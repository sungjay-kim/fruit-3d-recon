#!/usr/bin/env bash
# ArUco 마커 통합 3D reconstruction 파이프라인
#
# Usage:
#   bash bash/aruco_reconstruction.sh [SAMPLE] [BRIGHT_FACTOR] [ANGLE_DELTA] [GPU_IDX]
#
# 흐름:
#   1. generate_aruco_masks.py → sparse_masks/ (물체+마커 합성)
#   2. masks/ ↔ sparse_masks/ 교체 → sparse reconstruction (마커 feature 포함)
#   3. masks/ 복원
#   4. compute_aruco_scale.py → scale_factor.json
#   5. dense reconstruction (물체만 마스크)
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

# ── 전제 조건 확인 ──────────────────────────────────────────────────
if [ ! -d "${RUN_DIR}/images" ]; then
  echo "[ERROR] 이미지 폴더 없음: ${RUN_DIR}/images" >&2
  exit 1
fi
if [ ! -d "${RUN_DIR}/masks" ]; then
  echo "[ERROR] 물체 마스크 폴더 없음: ${RUN_DIR}/masks" >&2
  exit 1
fi

cd "$REPO_ROOT"

# ── micromamba 환경 활성화 (colmap, cv2, open3d 모두 포함) ────────────
MAMBA_BIN="$HOME/micromamba/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
if [ -x "$MAMBA_BIN" ]; then
  eval "$("$MAMBA_BIN" shell hook --shell bash)"
  micromamba activate colmap_env
fi

# ── Step 1: 원본 물체 마스크 보존 ─────────────────────────────────────
# masks/ 가 오염되지 않도록 원본을 object_masks/로 한 번만 복사
if [ ! -d "${RUN_DIR}/object_masks" ]; then
  log "[0/7] 원본 물체 마스크 보존 → object_masks/"
  cp -r "${RUN_DIR}/masks" "${RUN_DIR}/object_masks"
fi

# ── Step 1: ArUco 마스크 + 합성 sparse 마스크 생성 ──────────────────
T1=$(date +%s)
log "[1/7] ArUco 마스크 및 sparse_masks/ 생성"
# 항상 object_masks/를 기준으로 sparse_masks 생성 (오염 방지)
python3 scripts/generate_aruco_masks.py --sample "$SAMPLE" --object-masks-dir "${RUN_DIR}/object_masks"
elapsed $T1

# ── Step 2: sparse용 마스크 교체 ──────────────────────────────────────
log "[2/7] masks/ ← sparse_masks/ (sparse용 마커 포함 마스크)"
rm -rf "${RUN_DIR}/masks"
cp -r "${RUN_DIR}/sparse_masks" "${RUN_DIR}/masks"

# ── Step 3: Sparse Reconstruction ───────────────────────────────────
T3=$(date +%s)
log "[3/7] Sparse reconstruction (물체+마커 마스크 사용)"
bash bash/manual_relaxed_crosscam.sh "$SAMPLE" "$BRIGHT_FACTOR" "$ANGLE_DELTA"
elapsed $T3

# ── Step 4: dense용 마스크 복원 (물체만) ──────────────────────────────
log "[4/7] masks/ ← object_masks/ (dense용 물체만 마스크)"
rm -rf "${RUN_DIR}/masks"
cp -r "${RUN_DIR}/object_masks" "${RUN_DIR}/masks"

# ── Best model 선택 ──────────────────────────────────────────────────
log "Best sparse 모델 선택 중..."
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
log "  사용할 모델: ${best_model} (등록 이미지 수: ${best_count})"

# ── Step 5: Scale Factor 계산 ────────────────────────────────────────
T5=$(date +%s)
log "[5/6] ArUco scale factor 계산"
python3 scripts/compute_aruco_scale.py --sample "$SAMPLE" --model-idx "$best_model"
elapsed $T5

# ── Step 6: Dense Reconstruction ────────────────────────────────────
T6=$(date +%s)
log "[6/6] Dense reconstruction (물체만 마스크 사용)"
bash bash/dense.sh "$SAMPLE" "$best_model" "$GPU_IDX"
elapsed $T6

# ── Step 7: Scale 적용 ───────────────────────────────────────────────
T7=$(date +%s)
log "[7/6] Scale 적용 → fused_scaled.ply"
python3 scripts/apply_scale_to_ply.py --sample "$SAMPLE" --model-idx "$best_model"
elapsed $T7

# ── Step 8: 노이즈 제거 ──────────────────────────────────────────────
T8=$(date +%s)
log "[8/8] 노이즈 제거 → fused_scaled.ply (덮어쓰기)"
python3 scripts/remove_noise_ply.py --run-dir "${RUN_DIR}" --dense-subdir "dense/manual_relaxed_${best_model}"
elapsed $T8

log "완료! 총 소요시간: $(( $(date +%s) - TOTAL_START ))s"
log "  fused.ply        → runs/${SAMPLE}/dense/manual_relaxed_${best_model}/fused.ply"
log "  fused_scaled.ply → runs/${SAMPLE}/dense/manual_relaxed_${best_model}/fused_scaled.ply (노이즈 제거, 단위: m)"
log "  scale_factor     → runs/${SAMPLE}/scale_factor.json"
