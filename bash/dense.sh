#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash bash/dense.sh [SAMPLE_DIR] [MODEL_IDX] [GPU_IDX]
#     SAMPLE_DIR defaults to 20250804_1
#     MODEL_IDX  defaults to 0   (subfolder under sparse/manual_relaxed)
#     GPU_IDX    defaults to 0   (GPU index for PatchMatch)
SAMPLE="${1:-20250804_1}"
MODEL_IDX="${2:-0}"
GPU_IDX="${3:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
RUN_DIR="${RUNS_ROOT}/${SAMPLE}"

MODEL="${RUN_DIR}/sparse/manual_relaxed/${MODEL_IDX}"
DENSE_IMAGES_BASE="${RUN_DIR}/images_effective"
if [ ! -d "$DENSE_IMAGES_BASE" ]; then
  DENSE_IMAGES_BASE="${RUN_DIR}/images"
fi
DENSE_MASKS="${RUN_DIR}/masks"
DENSE="${RUN_DIR}/dense/manual_relaxed_${MODEL_IDX}"

START_TS=$(date +%s)

if [ ! -d "$MODEL" ]; then
  echo "Sparse model not found: $MODEL" >&2
  exit 1
fi

if [ ! -d "$DENSE_IMAGES_BASE" ]; then
  echo "Images folder not found: $DENSE_IMAGES_BASE" >&2
  exit 1
fi

mkdir -p "$DENSE"

echo "[1/4] Undistort images -> $DENSE"
colmap image_undistorter \
  --image_path "$DENSE_IMAGES_BASE" \
  --input_path "$MODEL" \
  --output_path "$DENSE" \
  --output_type COLMAP

echo "[2/4] PatchMatch Stereo (GPU: $GPU_IDX)"
colmap patch_match_stereo \
  --workspace_path "$DENSE" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency 0 \
  --PatchMatchStereo.max_image_size 1000 \
  --PatchMatchStereo.gpu_index "$GPU_IDX"

echo "[3/4] Stereo Fusion"
colmap stereo_fusion \
  --workspace_path "$DENSE" \
  --workspace_format COLMAP \
  --input_type photometric \
  --output_path "$DENSE/fused.ply"

echo "[4/4] Mask-based point filtering"
python3 - <<PY
from pathlib import Path
from PIL import Image
import numpy as np
import struct
import sys

dense = Path("${DENSE}")
masks_root = Path("${DENSE_MASKS}")
fused_path = dense / "fused.ply"

if not fused_path.exists():
    print("[ERROR] fused.ply not found")
    sys.exit(0)

# Build mask lookup: camera_id/image_name -> mask array
img_root = dense / "images"
masks = {}
for img_path in sorted(img_root.rglob("*.png")):
    rel = img_path.relative_to(img_root)
    # Try mask candidates
    for candidate in [
        masks_root / rel,
        masks_root / rel.with_suffix(rel.suffix + ".png"),
        masks_root / rel.parent / (rel.name + ".png"),
    ]:
        if candidate.exists():
            masks[str(rel)] = np.array(Image.open(candidate).convert("L")) > 128
            break

if not masks:
    print("No masks found — skipping point filtering")
    sys.exit(0)

# Read sparse model to get camera params and image poses
sparse_dir = dense / "sparse"
try:
    sys.path.insert(0, "${REPO_ROOT}/third_party/colmap_python")
    from read_write_model import read_model, qvec2rotmat
    cameras, images, _ = read_model(str(sparse_dir), ext=".bin")
except Exception as e:
    print(f"Cannot read sparse model for filtering: {e}")
    print("Skipping mask-based filtering")
    sys.exit(0)

# For each 3D point in fused.ply, project into images and check mask
# This is expensive, so use a simpler approach:
# Re-read fused.ply, project each point to ALL images, keep only if
# majority of visible views have the point inside the mask.

import open3d as o3d

pcd = o3d.io.read_point_cloud(str(fused_path))
pts = np.asarray(pcd.points)
colors = np.asarray(pcd.colors) if pcd.has_colors() else None

if len(pts) == 0:
    print("fused.ply is empty")
    sys.exit(0)

# Build projection matrices
proj_data = []  # (K, R, t, w, h, rel_path)
for img_id, img in images.items():
    cam = cameras[img.camera_id]
    fx, fy, cx, cy = cam.params[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R = qvec2rotmat(img.qvec)
    t = img.tvec
    rel = img.name  # e.g. "1/CAM#1_xxx.png"
    if rel in masks:
        proj_data.append((K, R, t, cam.width, cam.height, rel))

print(f"Projecting {len(pts)} points into {len(proj_data)} masked views...")

# For each point, count how many views see it inside vs outside the mask.
# Keep points where majority of visible views have it inside the mask.
n_pts = len(pts)
inside_count = np.zeros(n_pts, dtype=np.int32)
visible_count = np.zeros(n_pts, dtype=np.int32)

for K, R, t, w, h, rel in proj_data:
    mask = masks[rel]
    pts_cam = (R @ pts.T).T + t
    z = pts_cam[:, 2]
    valid_z = z > 0
    uv = (K @ pts_cam.T).T
    u = (uv[:, 0] / (uv[:, 2] + 1e-9)).astype(int)
    v = (uv[:, 1] / (uv[:, 2] + 1e-9)).astype(int)
    in_bounds = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    idx = np.where(in_bounds)[0]
    if len(idx) > 0:
        uu = u[idx].clip(0, w-1)
        vv = v[idx].clip(0, h-1)
        visible_count[idx] += 1
        inside_count[idx[mask[vv, uu]]] += 1

# Keep if majority of visible views have point inside mask
has_votes = visible_count > 0
inside_ratio = np.zeros(n_pts, dtype=np.float32)
inside_ratio[has_votes] = inside_count[has_votes] / visible_count[has_votes]
keep = inside_ratio > 0.5  # majority rule

kept = keep.sum()
removed = n_pts - kept
print(f"Mask filtering (majority vote): kept {kept}, removed {removed} ({removed/n_pts*100:.1f}%)")

if kept > 0 and removed > 0:
    pcd_filtered = pcd.select_by_index(np.where(keep)[0].tolist())
    o3d.io.write_point_cloud(str(fused_path), pcd_filtered)
    print(f"Overwrote {fused_path} with {kept} points")
elif kept == 0:
    print("[WARN] All points filtered — keeping original fused.ply")
PY

END_TS=$(date +%s)
echo "Total dense pipeline time: $((END_TS-START_TS)) seconds"
echo "Dense reconstruction complete: $DENSE/fused.ply"
