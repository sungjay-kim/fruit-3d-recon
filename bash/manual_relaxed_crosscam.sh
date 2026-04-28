#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash bash/manual_relaxed_crosscam.sh [SAMPLE_DIR] [BRIGHT_FACTOR] [ANGLE_DELTA]
#     SAMPLE_DIR defaults to 20250804_1
#     BRIGHT_FACTOR: brightness gain for cam2 (default: 1.0; set >1.0 to enable)
#     ANGLE_DELTA: degrees for neighbor pairing across/all cams (default: 20)
SAMPLE="${1:-20250804_1}"
BRIGHT_FACTOR="${2:-1.0}"
ANGLE_DELTA="${3:-20}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
RUN_DIR="${RUNS_ROOT}/${SAMPLE}"

DB="${RUN_DIR}/database/manual_relaxed.db"
IMAGES="${RUN_DIR}/images"
MASKS="${RUN_DIR}/masks"
OUT="${RUN_DIR}/sparse/manual_relaxed"
WORK="${RUN_DIR}/auto_workspace"
LOGS="${RUN_DIR}/logs"
BRIGHT_CAM2_DIR="${RUN_DIR}/images/2_bright"
EFFECTIVE_IMAGES="$IMAGES"

if [ ! -d "$IMAGES" ]; then
  echo "Image folder not found: $IMAGES" >&2
  exit 1
fi

first_image=$(find "$IMAGES" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print -quit)
if [ -z "$first_image" ]; then
  echo "No images found under $IMAGES" >&2
  exit 1
fi

mkdir -p "$WORK" "$OUT" "$LOGS" "$(dirname "$DB")"
rm -f "$DB"

# Brighten cam2 into images/2_bright if requested; otherwise use existing images/2.
if [ -d "${IMAGES}/2" ] && [ "$(printf '%.3f' "$BRIGHT_FACTOR")" != "1.000" ]; then
  mkdir -p "$BRIGHT_CAM2_DIR"
  RUN_DIR="$RUN_DIR" BRIGHT_CAM2_DIR="$BRIGHT_CAM2_DIR" BRIGHT_FACTOR="$BRIGHT_FACTOR" python3 - <<'PY'
import os
from pathlib import Path
from PIL import Image, ImageEnhance

run_dir = Path(os.environ["RUN_DIR"])
src_dir = run_dir / "images" / "2"
dst_dir = Path(os.environ["BRIGHT_CAM2_DIR"])
factor = float(os.environ["BRIGHT_FACTOR"])

if src_dir.exists():
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(src_dir.glob("*.png")):
        with Image.open(img_path).convert("RGB") as img:
            bright = ImageEnhance.Brightness(img).enhance(factor)
            bright.save(dst_dir / img_path.name)
PY
  # Build a temp tree with 1,2,3 where 2 holds the bright copies
  EFFECTIVE_IMAGES="${RUN_DIR}/images_effective"
  rm -rf "$EFFECTIVE_IMAGES"
  mkdir -p "$EFFECTIVE_IMAGES"/{1,2,3}
  cp -L -r "${IMAGES}/1/." "${EFFECTIVE_IMAGES}/1/"
  cp -L -r "${IMAGES}/3/." "${EFFECTIVE_IMAGES}/3/"
  cp -L -r "${BRIGHT_CAM2_DIR}/." "${EFFECTIVE_IMAGES}/2/"
fi

colmap feature_extractor \
  --database_path "$DB" \
  --image_path "$EFFECTIVE_IMAGES" \
  --ImageReader.mask_path "$MASKS" \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.single_camera_per_folder 1 \
  --ImageReader.single_camera 0 \
  --FeatureExtraction.num_threads 4 \
  --FeatureExtraction.use_gpu 1 \
  --SiftExtraction.max_num_features 8000 \
  --SiftExtraction.peak_threshold 0.005

img_count=$(sqlite3 "$DB" "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='images';" || echo 0)
if [ "$img_count" -eq 0 ]; then
  echo "Feature extraction did not create an images table in $DB" >&2
  exit 1
fi

colmap sequential_matcher \
  --database_path "$DB" \
  --FeatureMatching.guided_matching 1 \
  --SequentialMatching.overlap 30 \
  --SequentialMatching.quadratic_overlap 1

# Generate cross-camera, wraparound angle neighbors
EFFECTIVE_IMAGES="$EFFECTIVE_IMAGES" ANGLE_DELTA="$ANGLE_DELTA" LOGS_DIR="$LOGS" python3 - <<'PY'
import os
import re
from pathlib import Path
from itertools import combinations

root = Path(os.environ["EFFECTIVE_IMAGES"])
delta = int(os.environ["ANGLE_DELTA"])
cam_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
angle_map = {}
for cam_dir in cam_dirs:
    cam = cam_dir.name
    entries = []
    for img_path in sorted(cam_dir.glob("*.png")):
        m = re.search(r'Degree-([0-9]+)', img_path.name)
        if not m:
            continue
        ang = int(m.group(1)) % 360
        entries.append((ang, f"{cam}/{img_path.name}"))
    angle_map[cam] = entries

pairs = set()
def ang_dist(a, b):
    d = (b - a) % 360
    return min(d, 360 - d)

for cam_a, entries_a in angle_map.items():
    for ang_a, name_a in entries_a:
        for cam_b, entries_b in angle_map.items():
            for ang_b, name_b in entries_b:
                if ang_dist(ang_a, ang_b) <= delta:
                    pair = tuple(sorted((name_a, name_b)))
                    pairs.add(pair)

out_path = Path(os.environ["LOGS_DIR"]) / "cross_cam_pairs.txt"
out_path.write_text("\n".join(f"{a} {b}" for a, b in sorted(pairs)) + "\n")
print(f"wrote {len(pairs)} cross-camera/angle pairs to {out_path}")
PY

colmap matches_importer \
  --database_path "$DB" \
  --match_list_path "${LOGS}/cross_cam_pairs.txt" \
  --match_type pairs \
  --FeatureMatching.use_gpu 1 \
  --FeatureMatching.guided_matching 1 \
  --FeatureMatching.max_num_matches 2048

colmap mapper \
  --database_path "$DB" \
  --image_path "$EFFECTIVE_IMAGES" \
  --output_path "$OUT" \
  --Mapper.init_min_num_inliers 15 \
  --Mapper.abs_pose_min_num_inliers 8 \
  --Mapper.min_num_matches 8 \
  --Mapper.ba_refine_principal_point 0

# ─── Fallback: if registered image count is too low, retry with aggressive settings ───
TOTAL_IMAGES=$(find "$EFFECTIVE_IMAGES" -type f \( -iname '*.png' -o -iname '*.jpg' \) | wc -l)
REGISTRATION_THRESHOLD=$(( TOTAL_IMAGES / 2 ))  # 50% minimum

best_reg=0
for model_dir in "${OUT}"/*/; do
  [ -d "$model_dir" ] || continue
  count=$(colmap model_analyzer --path "$model_dir" 2>/dev/null \
    | grep -iE "registered (images|frames)" | awk '{print $NF}' | head -1)
  count="${count:-0}"
  [ "$count" -gt "$best_reg" ] && best_reg=$count
done

if [ "$best_reg" -lt "$REGISTRATION_THRESHOLD" ]; then
  echo "[fallback] Only ${best_reg}/${TOTAL_IMAGES} images registered (<${REGISTRATION_THRESHOLD}). Retrying with aggressive settings..."
  rm -rf "$OUT" "$DB" "${DB}-shm" "${DB}-wal"
  mkdir -p "$OUT" "$(dirname "$DB")"

  colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$EFFECTIVE_IMAGES" \
    --ImageReader.mask_path "$MASKS" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera_per_folder 1 \
    --ImageReader.single_camera 0 \
    --FeatureExtraction.num_threads 4 \
    --FeatureExtraction.use_gpu 1 \
    --SiftExtraction.max_num_features 16000 \
    --SiftExtraction.peak_threshold 0.0033 \
    --SiftExtraction.first_octave -1

  colmap exhaustive_matcher \
    --database_path "$DB" \
    --FeatureMatching.guided_matching 1 \
    --FeatureMatching.use_gpu 1 \
    --FeatureMatching.max_num_matches 8192

  colmap mapper \
    --database_path "$DB" \
    --image_path "$EFFECTIVE_IMAGES" \
    --output_path "$OUT" \
    --Mapper.init_min_num_inliers 8 \
    --Mapper.abs_pose_min_num_inliers 4 \
    --Mapper.min_num_matches 3 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.init_min_tri_angle 2 \
    --Mapper.multiple_models 0 \
    --Mapper.abs_pose_min_inlier_ratio 0.05 \
    --Mapper.abs_pose_max_error 20 \
    --Mapper.max_reg_trials 10 \
    --Mapper.init_max_reg_trials 20 \
    --Mapper.filter_max_reproj_error 8 \
    --Mapper.tri_min_angle 0.3 \
    --Mapper.filter_min_tri_angle 0.3 || true
fi
