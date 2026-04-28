"""
Scale Factor를 PLY 파일에 적용하는 스크립트

입력:
  runs/{SAMPLE}/dense/manual_relaxed_0/fused.ply
  runs/{SAMPLE}/scale_factor.json

출력:
  runs/{SAMPLE}/dense/manual_relaxed_0/fused_scaled.ply  (단위: m)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def apply_scale(ply_path: Path, scale_factor: float, out_path: Path):
    print(f"PLY 로드: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_pts = len(pcd.points)
    print(f"  포인트 수: {n_pts:,}")

    # scale 적용 (원점 기준)
    center = pcd.get_center()
    pcd.scale(scale_factor, center=np.zeros(3))

    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"저장: {out_path}")

    # 크기 확인
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    print(f"  크기 (m): {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="PLY에 scale factor 적용")
    parser.add_argument("--sample", required=True, help="샘플 이름 (예: 20260318_test)")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--model-idx", default=0, type=int, help="dense 모델 인덱스 (기본: 0)")
    parser.add_argument("--scale-factor", default=None, type=float,
                        help="직접 지정 (기본: scale_factor.json에서 읽음)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    runs_root = Path(args.runs_root) if args.runs_root else script_dir / "runs"
    run_dir = runs_root / args.sample

    # Scale factor 결정
    if args.scale_factor is not None:
        scale_factor = args.scale_factor
        print(f"Scale factor (직접 지정): {scale_factor:.4f}")
    else:
        sf_path = run_dir / "scale_factor.json"
        if not sf_path.exists():
            print(f"[ERROR] scale_factor.json 없음: {sf_path}")
            sys.exit(1)
        with open(sf_path) as f:
            sf_data = json.load(f)
        scale_factor = sf_data["scale_factor"]
        print(f"Scale factor (JSON): {scale_factor:.6f} {sf_data.get('unit', 'm')}/COLMAP_unit")

    # PLY 경로
    dense_dir = run_dir / "dense" / f"manual_relaxed_{args.model_idx}"
    ply_path = dense_dir / "fused.ply"
    if not ply_path.exists():
        print(f"[ERROR] fused.ply 없음: {ply_path}")
        sys.exit(1)

    out_path = dense_dir / "fused_scaled.ply"
    apply_scale(ply_path, scale_factor, out_path)


if __name__ == "__main__":
    main()
