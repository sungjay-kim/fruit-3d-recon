import argparse
import json
import os
import re
import shutil
import tempfile
import zipfile
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from sam3.model_builder import (
    build_sam3_multiplex_video_predictor,
    build_sam3_video_predictor,
    download_ckpt_from_hf,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (239, 71, 111),
    (17, 138, 178),
    (6, 214, 160),
    (255, 209, 102),
    (107, 70, 193),
    (255, 127, 14),
    (69, 123, 157),
    (214, 40, 40),
    (7, 59, 76),
    (146, 43, 62),
]


def _natural_sort_key(path: Path) -> List:
    m = re.search(r"\(Degree-(\d+)\)", path.name)
    if m:
        return [int(m.group(1)), path.name]
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def _prepare_frames_from_directory(
    frame_root: Path,
    temp_root: Path,
    max_frames: Optional[int],
    source_label: str,
) -> Tuple[Path, List[Path], List[str]]:
    image_files = sorted(
        [p for p in frame_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS],
        key=_natural_sort_key,
    )
    if len(image_files) == 0:
        raise ValueError(f"No images found in {source_label}")

    if max_frames is not None:
        image_files = image_files[:max_frames]

    frames_dir = temp_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: List[Path] = []
    frame_names: List[str] = []
    for idx, src in enumerate(image_files):
        dst = frames_dir / f"{idx:05d}{src.suffix.lower()}"
        shutil.copy(src, dst)
        frame_paths.append(dst)
        frame_names.append(src.name)

    return frames_dir, frame_paths, frame_names


def _prepare_frames_from_zip(
    zip_path: Path, temp_root: Path, max_frames: Optional[int]
) -> Tuple[Path, List[Path], List[str]]:
    extract_root = temp_root / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_root)

    return _prepare_frames_from_directory(
        extract_root, temp_root, max_frames, source_label=str(zip_path)
    )


def _prepare_frames_from_dir_path(
    frames_path: Path, temp_root: Path, max_frames: Optional[int]
) -> Tuple[Path, List[Path], List[str]]:
    return _prepare_frames_from_directory(
        frames_path, temp_root, max_frames, source_label=str(frames_path)
    )


def _save_overlay(
    frame_path: Path,
    masks: np.ndarray,
    output_path: Path,
    palette: Sequence[Tuple[int, int, int]],
    alpha: float = 0.5,
) -> None:
    frame = Image.open(frame_path).convert("RGB")
    canvas = np.array(frame)
    if masks.size == 0:
        Image.fromarray(canvas).save(output_path)
        return

    if masks.shape[1:] != canvas.shape[:2]:
        raise RuntimeError(
            f"Mask shape {masks.shape[1:]} does not match frame size {canvas.shape[:2]}"
        )

    for color, mask in zip(cycle(palette), masks):
        if not mask.any():
            continue
        color_arr = np.array(color, dtype=np.uint8)
        canvas[mask] = (
            canvas[mask].astype(np.float32) * (1.0 - alpha)
            + color_arr.astype(np.float32) * alpha
        ).astype(np.uint8)

    Image.fromarray(canvas).save(output_path)


def _save_binary_masks(
    masks: np.ndarray,
    obj_ids: Sequence[int],
    mask_dir: Path,
    frame_name: str,
) -> List[str]:
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_paths: List[str] = []
    stem = Path(frame_name).stem
    for obj_id, mask in zip(obj_ids, masks):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_path = mask_dir / f"{stem}_obj{obj_id}.png"
        mask_img.save(mask_path)
        mask_paths.append(mask_path.as_posix())
    return mask_paths


def _resolve_checkpoint(args: argparse.Namespace) -> Optional[Path]:
    if args.checkpoint:
        return Path(args.checkpoint)
    env_ckpt = os.getenv("SAM3_CHECKPOINT")
    if env_ckpt:
        return Path(env_ckpt)
    if args.allow_hf_download:
        downloaded = Path(download_ckpt_from_hf(version=args.model_version))
        return downloaded
    raise SystemExit(
        "Please pass --checkpoint or set SAM3_CHECKPOINT. "
        "Use --allow-hf-download to pull from Hugging Face instead."
    )


def _infer_model_version(checkpoint_path: Optional[Path], explicit: Optional[str]) -> str:
    if explicit and explicit != "auto":
        return explicit
    if checkpoint_path is not None and "3.1" in checkpoint_path.name:
        return "sam3.1"
    return "sam3"


def _patch_multiplex_init_state(predictor) -> None:
    # Upstream Sam3MultiplexTrackingWithInteractivity.init_state does not accept
    # `offload_state_to_cpu`, but Sam3BasePredictor.start_session always forwards it.
    # Wrap init_state to drop the unsupported kwarg.
    model = predictor.model
    orig_init_state = model.init_state

    def init_state(*args, **kwargs):
        kwargs.pop("offload_state_to_cpu", None)
        return orig_init_state(*args, **kwargs)

    model.init_state = init_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 video segmentation on a preloaded zip (or directory) of frames using a text prompt."
        )
    )
    parser.add_argument(
        "--zip-path",
        help="Path to a single .zip file containing ordered JPG/PNG frames.",
    )
    parser.add_argument(
        "--zip-dir",
        help="Directory containing multiple .zip archives to process in batch.",
    )
    parser.add_argument(
        "--frames-path",
        help="Path to a directory of ordered JPG/PNG frames (alternative to zip inputs).",
    )
    parser.add_argument(
        "--frames-dir",
        help="Directory containing multiple frame folders to process in batch.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt to ground (e.g. 'the red player').",
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to a SAM3 checkpoint. If omitted, SAM3_CHECKPOINT or Hugging Face is used.",
    )
    parser.add_argument(
        "--allow-hf-download",
        action="store_true",
        help="Allow downloading the checkpoint from Hugging Face if no local path is set.",
    )
    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help="Index of the frame to attach the text prompt to (default: 0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the number of frames loaded from the archive.",
    )
    parser.add_argument(
        "--direction",
        choices=["both", "forward", "backward"],
        default="both",
        help="Propagation direction for tracking results across the clip.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids to use (defaults to the current CUDA device).",
    )
    parser.add_argument(
        "--output-dir",
        default="demo_runs/text_prompt",
        help="Directory to store predictions; batch mode creates a subfolder per zip.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        choices=["json", "overlays", "masks"],
        help="Select which outputs to write (default json unless legacy flags used).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip any input whose output directory already exists.",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="[Deprecated] Save color overlays (use --outputs overlays instead).",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="[Deprecated] Save binary mask PNGs (use --outputs masks instead).",
    )
    parser.add_argument(
        "--model-version",
        choices=["auto", "sam3", "sam3.1"],
        default="auto",
        help="Which model builder to use. 'auto' infers from the checkpoint filename.",
    )
    parser.add_argument(
        "--use-fa3",
        action="store_true",
        help="Enable Flash Attention 3 for SAM 3.1 (requires Hopper GPU + flash-attn-3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("SAM3 video inference requires a CUDA-enabled GPU.")

    outputs_selected = set(args.outputs) if args.outputs else {"json"}
    if args.save_overlays:
        outputs_selected.add("overlays")
    if args.save_masks:
        outputs_selected.add("masks")
    if not outputs_selected:
        raise SystemExit("No outputs selected. Use --outputs to choose desired results.")

    targets: List[Tuple[str, Path]] = []
    if args.zip_path:
        zip_path = Path(args.zip_path).expanduser()
        if not zip_path.exists():
            raise SystemExit(f"Zip file not found: {zip_path}")
        targets.append(("zip", zip_path))
    if args.zip_dir:
        zip_dir = Path(args.zip_dir).expanduser()
        if not zip_dir.exists():
            raise SystemExit(f"Zip directory not found: {zip_dir}")
        targets.extend(("zip", z) for z in sorted(zip_dir.glob("*.zip")))
    if args.frames_path:
        frames_path = Path(args.frames_path).expanduser()
        if not frames_path.exists():
            raise SystemExit(f"Frames directory not found: {frames_path}")
        if not frames_path.is_dir():
            raise SystemExit(f"Frames path is not a directory: {frames_path}")
        targets.append(("frames", frames_path))
    if args.frames_dir:
        frames_dir = Path(args.frames_dir).expanduser()
        if not frames_dir.exists():
            raise SystemExit(f"Frames directory not found: {frames_dir}")
        if not frames_dir.is_dir():
            raise SystemExit(f"Frames path is not a directory: {frames_dir}")
        frame_subdirs = [p for p in sorted(frames_dir.iterdir()) if p.is_dir()]
        if not frame_subdirs:
            raise SystemExit(
                f"No subdirectories found in {frames_dir} to use as frame folders."
            )
        targets.extend(("frames", folder) for folder in frame_subdirs)
    if not targets:
        raise SystemExit(
            "Provide --zip-path/--zip-dir or --frames-path/--frames-dir to select inputs."
        )

    checkpoint_path = _resolve_checkpoint(args)
    gpus_to_use = None
    if args.gpus:
        gpus_to_use = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    if checkpoint_path is None:
        raise SystemExit(
            "Checkpoint path was not resolved. Provide --checkpoint or set SAM3_CHECKPOINT."
        )

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    explicit_batch = bool(args.zip_dir) or bool(args.frames_dir)
    multiple_runs = len(targets) > 1 or explicit_batch

    jobs: List[Tuple[str, Path, Path]] = []
    for input_type, input_path in targets:
        run_name = input_path.stem if input_type == "zip" else input_path.name
        target_output = out_base / run_name if multiple_runs else out_base
        if (
            args.skip_existing
            and target_output.exists()
            and any(target_output.iterdir())
        ):
            print(
                f"Skipping {input_path} because {target_output} already exists "
                "(remove it or omit --skip-existing to reprocess)."
            )
            continue
        jobs.append((input_type, input_path, target_output))

    if not jobs:
        print("No inputs to process (all skipped).")
        return

    model_version = _infer_model_version(checkpoint_path, args.model_version)
    if model_version == "sam3.1":
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_path),
            use_fa3=args.use_fa3,
            use_rope_real=args.use_fa3,
        )
        _patch_multiplex_init_state(predictor)
    else:
        predictor = build_sam3_video_predictor(
            checkpoint_path=str(checkpoint_path),
            gpus_to_use=gpus_to_use,
        )

    try:
        for input_type, input_path, target_output in jobs:
            target_output.mkdir(parents=True, exist_ok=True)
            print(f"Processing {input_path} ({input_type}) -> {target_output}")
            _process_single_input(
                input_type=input_type,
                source_path=input_path,
                output_dir=target_output,
                args=args,
                outputs_selected=outputs_selected,
                predictor=predictor,
            )
    finally:
        predictor.shutdown()


def _process_single_input(
    input_type: str,
    source_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    outputs_selected: set,
    predictor,
) -> None:
    overlay_dir = output_dir / "overlays" if "overlays" in outputs_selected else None
    mask_dir = output_dir / "masks" if "masks" in outputs_selected else None

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            if input_type == "zip":
                frames_dir, frame_paths, frame_names = _prepare_frames_from_zip(
                    source_path, Path(tmpdir), args.max_frames
                )
            else:
                frames_dir, frame_paths, frame_names = _prepare_frames_from_dir_path(
                    source_path, Path(tmpdir), args.max_frames
                )
        except ValueError as exc:
            print(
                f"Skipping {source_path} because no images were found ({exc}). "
                f"{output_dir} will remain empty."
            )
            return
        if args.prompt_frame >= len(frame_paths):
            raise SystemExit(
                f"Prompt frame {args.prompt_frame} is out of range for "
                f"{len(frame_paths)} frames."
            )

        session_id = predictor.handle_request(
            {"type": "start_session", "resource_path": str(frames_dir)}
        )["session_id"]

        frame_summaries: Dict[int, Dict] = {}

        def store_outputs(frame_idx: int, outputs: Dict) -> None:
            if outputs is None:
                return

            object_ids = [int(o) for o in outputs["out_obj_ids"].tolist()]
            boxes = np.asarray(outputs["out_boxes_xywh"]).tolist()
            scores = [float(s) for s in np.asarray(outputs["out_probs"]).tolist()]
            frame_name = frame_names[frame_idx]
            entry = {
                "frame_index": int(frame_idx),
                "frame_name": frame_name,
                "object_ids": object_ids,
                "scores": scores,
                "boxes_xywh": boxes,
            }

            masks = np.asarray(outputs["out_binary_masks"])
            if overlay_dir is not None:
                overlay_dir.mkdir(parents=True, exist_ok=True)
                overlay_path = overlay_dir / f"{Path(frame_name).stem}_overlay.png"
                _save_overlay(
                    frame_paths[frame_idx], masks, overlay_path, COLOR_PALETTE
                )
                entry["overlay"] = overlay_path.as_posix()
            if mask_dir is not None and masks.size > 0:
                entry["mask_paths"] = _save_binary_masks(
                    masks, object_ids, mask_dir, frame_name
                )

            frame_summaries[frame_idx] = entry

        prompt_result = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": args.prompt_frame,
                "text": args.prompt,
            }
        )
        store_outputs(prompt_result["frame_index"], prompt_result["outputs"])

        stream_request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": args.direction,
            "start_frame_index": args.prompt_frame,
            "max_frame_num_to_track": len(frame_paths),
        }
        for result in predictor.handle_stream_request(stream_request):
            if not result:
                continue
            frame_idx = result.get("frame_index")
            outputs = result.get("outputs")
            if frame_idx is None or outputs is None:
                continue
            store_outputs(frame_idx, outputs)

        predictor.handle_request({"type": "close_session", "session_id": session_id})

    ordered_frames = [
        frame_summaries[idx] for idx in sorted(frame_summaries.keys())
    ]
    summary = {
        "input_type": input_type,
        "input_path": source_path.as_posix(),
        "prompt": args.prompt,
        "prompt_frame": args.prompt_frame,
        "propagation_direction": args.direction,
        "num_frames_used": len(frame_paths),
        "results": ordered_frames,
    }
    if input_type == "zip":
        summary["zip_path"] = source_path.as_posix()
    else:
        summary["frames_path"] = source_path.as_posix()

    if "json" in outputs_selected:
        summary_path = output_dir / "predictions.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {len(ordered_frames)} frame results to {summary_path}")
    else:
        print(
            f"Processed {len(ordered_frames)} frames for {source_path} (JSON disabled)."
        )

    if overlay_dir and overlay_dir.exists():
        print(f"Overlays saved to {overlay_dir}")
    if mask_dir and mask_dir.exists():
        print(f"Binary masks saved to {mask_dir}")


if __name__ == "__main__":
    main()
