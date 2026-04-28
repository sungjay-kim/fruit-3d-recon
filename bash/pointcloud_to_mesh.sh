#!/usr/bin/env bash
set -euo pipefail

# Convert an existing dense fused point cloud into mesh representations.
# Usage:
#   bash bash/pointcloud_to_mesh.sh [SAMPLE_ID] [dense_subdir]
#     SAMPLE_ID    run folder name under runs/ (default: 20250804_1)
#     dense_subdir relative path under dense/ containing fused.ply. If omitted,
#                  the newest fused.ply inside dense/ is selected automatically.

SAMPLE="${1:-20250804_1}"
EXPLICIT_DENSE="${2:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${REPO_ROOT}/runs/${SAMPLE}"
DENSE_ROOT="${RUN_DIR}/dense"
COLMAP_BIN="${COLMAP_BIN:-colmap}"
VOLUME_PY="${REPO_ROOT}/scripts/mesh_volume.py"
GLOBAL_VOLUME_FILE="${GLOBAL_VOLUME_FILE:-${REPO_ROOT}/volumes.json}"
PREPROCESS_PY="${REPO_ROOT}/scripts/preprocess_pointcloud_for_meshing.py"
PREPROCESS_FUSED="${PREPROCESS_FUSED:-1}"
PREPROCESS_BLACK_THRESHOLD="${PREPROCESS_BLACK_THRESHOLD:-10}"
RUN_POISSON="${RUN_POISSON:-1}"
RUN_DELAUNAY="${RUN_DELAUNAY:-0}"
POISSON_DEPTH="${POISSON_DEPTH:-9}"
POISSON_POINT_WEIGHT="${POISSON_POINT_WEIGHT:-1.0}"
POISSON_THREADS="${POISSON_THREADS:--1}"
POISSON_TRIM="${POISSON_TRIM:-6}"
POISSON_COLOR="${POISSON_COLOR:-32}"
POISSON_ENABLE_COLOR="${POISSON_ENABLE_COLOR:-1}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

choose_fused_from_logs() {
  local log_dir="${RUN_DIR}/logs"
  [ -d "$log_dir" ] || return 1
  local latest_log
  latest_log="$(ls -1t "${log_dir}"/pipeline_*.log 2>/dev/null | head -n 1 || true)"
  [ -n "$latest_log" ] || return 1
  local line
  line="$(grep -E 'Dense output:' "$latest_log" | tail -n 1 || true)"
  [ -n "$line" ] || return 1
  line="${line##*: }"
  [ -n "$line" ] || return 1
  echo "$line"
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

command -v "$COLMAP_BIN" >/dev/null 2>&1 || [ -x "$COLMAP_BIN" ] || fail "COLMAP binary not found: $COLMAP_BIN (set COLMAP_BIN env var or ensure 'colmap' is on PATH)"
[ "$PREPROCESS_FUSED" -eq 0 ] || [ -f "$PREPROCESS_PY" ] || fail "Preprocess script not found at $PREPROCESS_PY"
[ -d "$RUN_DIR" ] || fail "Sample directory not found: $RUN_DIR"
[ -d "$DENSE_ROOT" ] || fail "Dense directory not found: $DENSE_ROOT"

select_fused() {
  local explicit="$1"
  if [ -n "$explicit" ]; then
    local fused_path="${DENSE_ROOT}/${explicit}/fused.ply"
    [ -f "$fused_path" ] || fail "Explicit fused point cloud not found: $fused_path"
    echo "$fused_path"
    return
  fi
  local log_choice
  log_choice="$(choose_fused_from_logs || true)"
  if [ -n "$log_choice" ] && [ -f "$log_choice" ]; then
    echo "$log_choice"
    return
  fi
  mapfile -t candidates < <(find "$DENSE_ROOT" -mindepth 1 -maxdepth 4 -type f -name 'fused.ply' | sort)
  local count="${#candidates[@]}"
  if [ "$count" -eq 0 ]; then
    fail "No fused.ply files detected under $DENSE_ROOT"
  fi
  if [ "$count" -eq 1 ]; then
    echo "${candidates[0]}"
    return
  fi
  local newest_path=""
  local newest_ts=0
  for cand in "${candidates[@]}"; do
    local ts
    ts="$(stat -c %Y "$cand")"
    if [ "$ts" -gt "$newest_ts" ]; then
      newest_ts="$ts"
      newest_path="$cand"
    fi
  done
  log "Multiple fused.ply files found; selecting newest: $newest_path" >&2
  echo "$newest_path"
}

FUSED_PATH="$(select_fused "$EXPLICIT_DENSE")"
[ -n "$FUSED_PATH" ] || fail "Unable to resolve fused point cloud path"
WORK_DIR="$(dirname "$FUSED_PATH")"
POISSON_MESH="${WORK_DIR}/meshed_poisson.ply"
POISSON_MESH_COLOR="${WORK_DIR}/meshed_poisson_color.ply"
DELAUNAY_MESH="${WORK_DIR}/meshed_delaunay.ply"
DENSE_SUBDIR="$(basename "$WORK_DIR")"
CLEANED_FUSED_PATH="${WORK_DIR}/fused_preprocessed.ply"
MESHER_INPUT_PATH="$FUSED_PATH"

remove_if_exists() {
  local target="$1"
  if [ -e "$target" ]; then
    rm -f "$target"
  fi
}

log "Sample: $SAMPLE"
log "Using fused point cloud: $FUSED_PATH"
log "Output directory: $WORK_DIR"

if [ "$PREPROCESS_FUSED" -eq 1 ]; then
  log "Preprocessing fused point cloud for meshing (black-point filter)..."
  remove_if_exists "$CLEANED_FUSED_PATH"
  python3 "$PREPROCESS_PY" \
    "$FUSED_PATH" "$CLEANED_FUSED_PATH" \
    --black-threshold "$PREPROCESS_BLACK_THRESHOLD"
  MESHER_INPUT_PATH="$CLEANED_FUSED_PATH"
  log "Using preprocessed point cloud: $MESHER_INPUT_PATH"
else
  log "Skipping preprocessing; meshing fused.ply directly"
fi

record_volume() {
  local mesh_path="$1"
  local label="$2"
  if [ ! -s "$mesh_path" ]; then
    log "Skipping volume computation for ${label} (missing mesh: $mesh_path)"
    return
  fi
  if [ -z "$GLOBAL_VOLUME_FILE" ]; then
    log "GLOBAL_VOLUME_FILE is empty; skipping volume recording"
    return
  fi
  if [ ! -f "$VOLUME_PY" ]; then
    log "Volume script not found at $VOLUME_PY"
    return
  fi
  local args=("$mesh_path" "-" "$label" "$SAMPLE" "$DENSE_SUBDIR" "$GLOBAL_VOLUME_FILE")
  if python3 "$VOLUME_PY" "${args[@]}"; then
    log "Recorded ${label} volume in $GLOBAL_VOLUME_FILE"
  else
    log "Volume computation failed for $mesh_path"
  fi
}

run_poisson_variant() {
  local output_path="$1"
  local color_value="$2"
  local label="$3"
  log "Running COLMAP Poisson mesher (${label})..."
  "$COLMAP_BIN" poisson_mesher \
    --input_path "$MESHER_INPUT_PATH" \
    --output_path "$output_path" \
    --PoissonMeshing.depth "$POISSON_DEPTH" \
    --PoissonMeshing.point_weight "$POISSON_POINT_WEIGHT" \
    --PoissonMeshing.num_threads "$POISSON_THREADS" \
    --PoissonMeshing.trim "$POISSON_TRIM" \
    --PoissonMeshing.color "$color_value"
  log "Poisson mesh (${label}) written to $output_path"
  record_volume "$output_path" "$label"
}

remove_if_exists "$POISSON_MESH"
remove_if_exists "$POISSON_MESH_COLOR"
remove_if_exists "$DELAUNAY_MESH"
if [ "$RUN_POISSON" -eq 1 ]; then
  run_poisson_variant "$POISSON_MESH" 0 "meshed_poisson"
  if [ "$POISSON_ENABLE_COLOR" -ne 0 ]; then
    run_poisson_variant "$POISSON_MESH_COLOR" "$POISSON_COLOR" "meshed_poisson_color"
  else
    log "Skipping color Poisson mesh (POISSON_ENABLE_COLOR=$POISSON_ENABLE_COLOR)"
  fi
else
  log "Skipping Poisson mesher (RUN_POISSON=$RUN_POISSON)"
fi

if [ "$RUN_DELAUNAY" -eq 1 ]; then
  log "Running COLMAP Delaunay mesher..."
  "$COLMAP_BIN" delaunay_mesher \
    --input_path "$WORK_DIR" \
    --input_type dense \
    --output_path "$DELAUNAY_MESH"
  log "Delaunay mesh written to $DELAUNAY_MESH"
  record_volume "$DELAUNAY_MESH" "meshed_delaunay"
else
  log "Skipping Delaunay mesher (RUN_DELAUNAY=$RUN_DELAUNAY)"
fi

log "Mesh conversion completed."
