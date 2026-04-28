#!/usr/bin/env python3
"""Copy images and SAM3 masks into the runs/ directory layout expected by aruco_reconstruction.sh.

Source layout:
    {merged_root}/{sample}/{cam}/*.png       # RGB frames per camera
    {mask_root}/{sample}_{cam}/masks/*.png   # SAM3 binary masks (preferred)
    {mask_root}/{sample}/{cam}/masks/*.png   # SAM3 binary masks (fallback)

Destination layout:
    {runs_root}/{sample}/images/{cam}/*.png
    {runs_root}/{sample}/masks/{cam}/*.png

Each mask is matched to its image by stem: {image_stem}_obj0.png in the source.
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged-root", type=Path, required=True,
                        help="Root containing {sample}/{cam}/*.png")
    parser.add_argument("--mask-root", type=Path, required=True,
                        help="Root containing SAM3 outputs ({sample}_{cam}/masks/ or {sample}/{cam}/masks/)")
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="Destination root for reconstruction runs")
    parser.add_argument("--cams", nargs="+", default=["1", "2", "3"],
                        help="Camera subdirectory names (default: 1 2 3)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing destination samples")
    args = parser.parse_args()

    cams = tuple(args.cams)
    samples = {}
    for entry in args.merged_root.iterdir():
        if not entry.is_dir():
            continue
        cam_dirs = [c for c in entry.iterdir() if c.is_dir() and c.name in cams]
        if cam_dirs:
            samples[entry.name] = {c.name for c in cam_dirs}

    for sample in sorted(samples):
        dest_run = args.runs_root / sample
        if dest_run.exists() and not args.overwrite:
            print(f"[skip] {sample} (already exists, use --overwrite to replace)")
            continue
        for sub in ("images", "masks"):
            for cam in cams:
                (dest_run / sub / cam).mkdir(parents=True, exist_ok=True)
        for cam in cams:
            src_img_dir = args.merged_root / sample / cam
            if not src_img_dir.exists():
                continue
            dest_img_dir = dest_run / "images" / cam
            image_files = [
                p for p in sorted(src_img_dir.iterdir())
                if p.is_file() and p.suffix.lower() == ".png"
            ]
            for img in image_files:
                shutil.copy2(img, dest_img_dir / img.name)
            src_mask_dir = args.mask_root / f"{sample}_{cam}" / "masks"
            if not src_mask_dir.exists():
                src_mask_dir = args.mask_root / sample / cam / "masks"
            if not src_mask_dir.exists():
                continue
            dest_mask_dir = dest_run / "masks" / cam
            for img in image_files:
                mask_src = src_mask_dir / f"{img.stem}_obj0.png"
                if mask_src.exists():
                    shutil.copy2(mask_src, dest_mask_dir / f"{img.name}.png")
        print(f"[ok]   {sample} → {dest_run}")


if __name__ == "__main__":
    main()
