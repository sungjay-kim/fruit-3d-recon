"""
Apply a scale factor to a PLY point cloud.

Inputs:
  runs/{SAMPLE}/dense/manual_relaxed_0/fused.ply
  runs/{SAMPLE}/scale_factor.json

Output:
  runs/{SAMPLE}/dense/manual_relaxed_0/fused_scaled.ply  (units: m)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def apply_scale(ply_path: Path, scale_factor: float, out_path: Path):
    print(f"Loading PLY: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_pts = len(pcd.points)
    print(f"  point count: {n_pts:,}")

    pcd.scale(scale_factor, center=np.zeros(3))

    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Saved: {out_path}")

    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    print(f"  size (m): {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Apply scale factor to a PLY point cloud")
    parser.add_argument("--sample", required=True, help="sample name (e.g. 20260318_test)")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--model-idx", default=0, type=int, help="dense model index (default: 0)")
    parser.add_argument("--scale-factor", default=None, type=float,
                        help="explicit scale factor (default: read from scale_factor.json)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    runs_root = Path(args.runs_root) if args.runs_root else script_dir / "runs"
    run_dir = runs_root / args.sample

    if args.scale_factor is not None:
        scale_factor = args.scale_factor
        print(f"Scale factor (explicit): {scale_factor:.4f}")
    else:
        sf_path = run_dir / "scale_factor.json"
        if not sf_path.exists():
            print(f"[ERROR] scale_factor.json not found: {sf_path}")
            sys.exit(1)
        with open(sf_path) as f:
            sf_data = json.load(f)
        scale_factor = sf_data["scale_factor"]
        print(f"Scale factor (JSON): {scale_factor:.6f} {sf_data.get('unit', 'm')}/COLMAP_unit")

    dense_dir = run_dir / "dense" / f"manual_relaxed_{args.model_idx}"
    ply_path = dense_dir / "fused.ply"
    if not ply_path.exists():
        print(f"[ERROR] fused.ply not found: {ply_path}")
        sys.exit(1)

    out_path = dense_dir / "fused_scaled.ply"
    apply_scale(ply_path, scale_factor, out_path)


if __name__ == "__main__":
    main()
