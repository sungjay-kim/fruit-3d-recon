#!/usr/bin/env bash
# ArUco-integrated 3D reconstruction pipeline
#
# Usage:
#   bash bash/aruco_reconstruction.sh [SAMPLE] [BRIGHT_FACTOR] [ANGLE_DELTA] [GPU_IDX]
#
# Flow:
#   1. generate_aruco_masks.py → sparse_masks/ (object + marker composite)
#   2. swap masks/ ↔ sparse_masks/ → sparse reconstruction (marker features included)
#   3. restore masks/
#   4. compute_aruco_scale.py → scale_factor.json
#   5. dense reconstruction (object-only masks)
#   6. apply_scale_to_ply.py → fused_scaled.ply
set -eo pipefail

SAMPLE="${1:-20260318_test}"
BRIGHT_FACTOR="${2:-1.0}"
ANGLE_DELTA="${3:-20}"
GPU_IDX="${4:-0,1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${REPO_ROOT}/runs/${SAMPLE}"
SPARSE_OUT="${RUN_DIR}/sparse/manual_relaxed"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
elapsed() { echo "  ⏱ $(( $(date +%s) - $1 ))s"; }
TOTAL_START=$(date +%s)

# ── Preconditions ───────────────────────────────────────────────────
if [ ! -d "${RUN_DIR}/images" ]; then
  echo "[ERROR] images folder not found: ${RUN_DIR}/images" >&2
  exit 1
fi
if [ ! -d "${RUN_DIR}/masks" ]; then
  echo "[ERROR] object masks folder not found: ${RUN_DIR}/masks" >&2
  exit 1
fi

cd "$REPO_ROOT"

# ── Activate micromamba env (colmap, cv2, open3d all included) ──────
MAMBA_BIN="$HOME/micromamba/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
if [ -x "$MAMBA_BIN" ]; then
  eval "$("$MAMBA_BIN" shell hook --shell bash)"
  micromamba activate colmap_env
fi

# ── Step 0: snapshot original object masks ──────────────────────────
# Keep masks/ pristine by copying it once into object_masks/.
if [ ! -d "${RUN_DIR}/object_masks" ]; then
  log "[0/7] Snapshot object masks → object_masks/"
  cp -r "${RUN_DIR}/masks" "${RUN_DIR}/object_masks"
fi

# ── Step 1: generate ArUco + composite sparse masks ─────────────────
T1=$(date +%s)
log "[1/7] Generate ArUco masks and sparse_masks/"
# Always derive sparse_masks from object_masks/ (avoid contamination).
python3 scripts/generate_aruco_masks.py --sample "$SAMPLE" --object-masks-dir "${RUN_DIR}/object_masks"
elapsed $T1

# ── Step 2: swap in sparse masks ────────────────────────────────────
log "[2/7] masks/ ← sparse_masks/ (object + marker masks for sparse)"
rm -rf "${RUN_DIR}/masks"
cp -r "${RUN_DIR}/sparse_masks" "${RUN_DIR}/masks"

# ── Step 3: Sparse Reconstruction ───────────────────────────────────
T3=$(date +%s)
log "[3/7] Sparse reconstruction (object + marker masks)"
bash bash/manual_relaxed_crosscam.sh "$SAMPLE" "$BRIGHT_FACTOR" "$ANGLE_DELTA"
elapsed $T3

# ── Step 4: restore object-only masks for dense ─────────────────────
log "[4/7] masks/ ← object_masks/ (object-only masks for dense)"
rm -rf "${RUN_DIR}/masks"
cp -r "${RUN_DIR}/object_masks" "${RUN_DIR}/masks"

# ── Best model selection ────────────────────────────────────────────
log "Selecting best sparse model..."
best_model=0
best_count=0
for model_dir in "${SPARSE_OUT}"/*/; do
  [ -d "$model_dir" ] || continue
  model_idx="$(basename "$model_dir")"
  count=$(colmap model_analyzer \
    --path "$model_dir" 2>/dev/null \
    | grep -i "registered images" | awk '{print $NF}' || echo 0)
  if [ "${count:-0}" -gt "$best_count" ]; then
    best_count=$count
    best_model=$model_idx
  fi
done
log "  selected model: ${best_model} (registered images: ${best_count})"

# ── Step 5: Compute scale factor ────────────────────────────────────
T5=$(date +%s)
log "[5/8] Compute ArUco scale factor"
python3 scripts/compute_aruco_scale.py --sample "$SAMPLE" --model-idx "$best_model"
elapsed $T5

# ── Step 6: Dense Reconstruction ────────────────────────────────────
T6=$(date +%s)
log "[6/8] Dense reconstruction (object-only masks)"
bash bash/dense.sh "$SAMPLE" "$best_model" "$GPU_IDX"
elapsed $T6

# ── Step 7: Apply scale ─────────────────────────────────────────────
T7=$(date +%s)
log "[7/8] Apply scale → fused_scaled.ply"
python3 scripts/apply_scale_to_ply.py --sample "$SAMPLE" --model-idx "$best_model"
elapsed $T7

# ── Step 8: Denoise ─────────────────────────────────────────────────
T8=$(date +%s)
log "[8/8] Denoise → overwrite fused_scaled.ply"
python3 scripts/remove_noise_ply.py --run-dir "${RUN_DIR}" --dense-subdir "dense/manual_relaxed_${best_model}"
elapsed $T8

log "Done! Total time: $(( $(date +%s) - TOTAL_START ))s"
log "  fused.ply        → runs/${SAMPLE}/dense/manual_relaxed_${best_model}/fused.ply"
log "  fused_scaled.ply → runs/${SAMPLE}/dense/manual_relaxed_${best_model}/fused_scaled.ply (denoised, units: m)"
log "  scale_factor     → runs/${SAMPLE}/scale_factor.json"
