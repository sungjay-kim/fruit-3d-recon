"""
Remove black/dark noise from fused_scaled.ply using:
1. Color brightness threshold (remove very dark points)
2. Statistical Outlier Removal (remove isolated floating points)

Output: fused_scaled_clean.ply (same directory as input)
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def remove_noise(
    ply_path: Path,
    brightness_thresh: float = 0.05,
    sor_nb: int = 20,
    sor_std: float = 2.0,
) -> Path:
    out_path = ply_path  # overwrite fused_scaled.ply in-place

    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_orig = len(pcd.points)

    # Step 1: color brightness filter
    colors = np.asarray(pcd.colors)
    brightness = colors.mean(axis=1)
    color_mask = brightness >= brightness_thresh
    pcd = pcd.select_by_index(np.where(color_mask)[0])
    n_after_color = len(pcd.points)

    # Step 2: Statistical Outlier Removal
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=sor_nb,
        std_ratio=sor_std,
    )
    n_final = len(pcd_clean.points)

    o3d.io.write_point_cloud(str(out_path), pcd_clean)

    print(f"  {run_label(ply_path)}")
    print(f"    Original : {n_orig:,}")
    print(f"    After color filter (brightness >= {brightness_thresh}): {n_after_color:,}  (-{n_orig - n_after_color:,})")
    print(f"    After SOR (nb={sor_nb}, std={sor_std}): {n_final:,}  (-{n_after_color - n_final:,})")
    print(f"    Total removed: {n_orig - n_final:,}  ({100*(n_orig - n_final)/n_orig:.1f}%)")
    print(f"    -> {out_path} (overwritten)")
    return out_path


def run_label(ply_path: Path) -> str:
    # .../runs/20260320_108/dense/manual_relaxed_0/fused_scaled.ply -> 20260320_108
    parts = ply_path.parts
    try:
        idx = parts.index("runs")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return ply_path.parent.name


def main():
    parser = argparse.ArgumentParser(description="Remove black noise from PLY files")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Single run directory (e.g. runs/20260320_108). "
                             "If omitted, processes all runs under --runs-root.")
    parser.add_argument("--runs-root", type=str, default=None,
                        help="Root directory containing run folders (default: <repo>/runs).")
    parser.add_argument("--dense-subdir", type=str, default="dense/manual_relaxed_0",
                        help="Subdirectory inside run dir that contains fused_scaled.ply")
    parser.add_argument("--brightness-thresh", type=float, default=0.05)
    parser.add_argument("--sor-nb", type=int, default=20)
    parser.add_argument("--sor-std", type=float, default=2.0)
    args = parser.parse_args()

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        if args.runs_root:
            runs_root = Path(args.runs_root)
        else:
            runs_root = Path(__file__).resolve().parent.parent / "runs"
        run_dirs = sorted([d for d in runs_root.iterdir() if d.is_dir()])

    processed, skipped = 0, 0
    for run_dir in run_dirs:
        ply = run_dir / args.dense_subdir / "fused_scaled.ply"
        if not ply.exists():
            skipped += 1
            continue
        remove_noise(
            ply,
            brightness_thresh=args.brightness_thresh,
            sor_nb=args.sor_nb,
            sor_std=args.sor_std,
        )
        processed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped (no fused_scaled.ply)")


if __name__ == "__main__":
    main()
