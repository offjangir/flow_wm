#!/usr/bin/env bash
#
# End-to-end: build data_wan/ for world-model training from a raw DROID
# dataset directory. Edit scripts/build_data_wan.config to set paths.
#
# Pipeline (each phase is idempotent — re-run skips finished scenes):
#
#   [1] Render DrRobot clips (action-conditioning video)
#         drrobot/render_droid_scenes.py
#         -> ${DATA_WAN}/clips/<scene>.mp4
#
#   [2] Extract AllTracker DENSE per-pixel tracks on real DROID camera video
#       (= track conditioning for the world model). Sparse tracks are also
#       written so prepare_data_wan.py's metadata.csv has its `tracks` column.
#         scripts/extract_alltracker.py --save_dense_tracks
#         -> ${DATA_WAN}/alltracker_dense_tracks/<scene>.npz   (dense)
#         -> ${DATA_WAN}/alltracker_tracks/<scene>.npz         (sparse)
#
#   [3] Symlink real DROID RGB into ${DATA_WAN}/videos/ + write metadata.csv
#         prepare_data_wan.py
#         -> ${DATA_WAN}/videos/<scene>.mp4
#         -> ${DATA_WAN}/metadata.csv  (video,prompt,render,tracks)
#
# Usage:
#   ./scripts/build_data_wan.sh                              # uses default config
#   ./scripts/build_data_wan.sh --config path/to/other.config
#   MAX_SCENES=2 ./scripts/build_data_wan.sh                 # env override
#
# Precedence: real env var > config file > script default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ─── Locate config ───────────────────────────────────────────────────────────
CONFIG_FILE="${SCRIPT_DIR}/build_data_wan.config"
if [[ "${1:-}" == "--config" && -n "${2:-}" ]]; then
  CONFIG_FILE="$2"
  shift 2
fi

# Snapshot env vars that should win over the config file. If they're set in
# the real environment, we restore them after sourcing the config.
_ENV_KEYS=(
  DROID_ROOT MODEL_PATH ALLTRACKER_ROOT DATA_WAN
  CAMERA_KEY DROID_SERIAL
  RENDER_FPS RENDER_BG
  TRACK_GRID_X TRACK_GRID_Y TRACK_IMAGE_SIZE
  PY MAX_SCENES SKIP_RENDER SKIP_TRACKS SKIP_METADATA
)
declare -A _ENV_SAVED=()
for k in "${_ENV_KEYS[@]}"; do
  if [[ -n "${!k+x}" ]]; then _ENV_SAVED[$k]="${!k}"; fi
done

if [[ -f "${CONFIG_FILE}" ]]; then
  echo "[config] sourcing ${CONFIG_FILE}"
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
else
  echo "[config] ${CONFIG_FILE} not found, using script defaults / env"
fi

for k in "${!_ENV_SAVED[@]}"; do
  printf -v "$k" '%s' "${_ENV_SAVED[$k]}"
done

# ─── Defaults (only used if neither env nor config set the value) ───────────
: "${DROID_ROOT:=${REPO_ROOT}/data/droid_10_demos}"
: "${MODEL_PATH:=${REPO_ROOT}/drrobot/output/franka_emika_panda_complement1}"
: "${DATA_WAN:=${REPO_ROOT}/data_wan}"
: "${CAMERA_KEY:=ext1}"
: "${DROID_SERIAL:=20103212}"
: "${ALLTRACKER_ROOT:=/data/yjangir/sidegig/alltracker}"
: "${PY:=python}"
: "${MAX_SCENES:=0}"
: "${RENDER_FPS:=match}"
: "${RENDER_BG:=black}"
: "${TRACK_GRID_X:=32}"
: "${TRACK_GRID_Y:=20}"
: "${TRACK_IMAGE_SIZE:=1024}"
: "${SKIP_RENDER:=0}"
: "${SKIP_TRACKS:=0}"
: "${SKIP_METADATA:=0}"

# Force a clean compute_90 arch list. On GH200 (sm_90) torch reports both
# 'sm_90' and 'sm_90a' in get_arch_list(); gsplat's CUDA-extension loader
# parses each entry as int(arch.removeprefix('sm_')) and dies on '90a'. Pinning
# to "9.0" keeps the JIT compile + .so reload happy without losing performance
# (sm_90a additions are unused by gsplat). Do not collapse to TORCH_CUDA_ARCH=9.0a.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# ─── Sanity ──────────────────────────────────────────────────────────────────
[[ -d "${DROID_ROOT}" ]] || { echo "ERROR: DROID_ROOT not a directory: ${DROID_ROOT}"; exit 1; }
if [[ "${SKIP_RENDER}" != "1" ]]; then
  [[ -d "${MODEL_PATH}" && -f "${MODEL_PATH}/cfg_args" ]] || {
    echo "ERROR: MODEL_PATH must be a DrRobot run dir containing cfg_args: ${MODEL_PATH}"
    exit 1
  }
fi
if [[ "${SKIP_TRACKS}" != "1" ]]; then
  [[ -d "${ALLTRACKER_ROOT}/nets" ]] || {
    echo "ERROR: ALLTRACKER_ROOT missing nets/ — clone https://github.com/aharley/alltracker into ${ALLTRACKER_ROOT}"
    exit 1
  }
fi

mkdir -p \
  "${DATA_WAN}/clips" \
  "${DATA_WAN}/videos" \
  "${DATA_WAN}/alltracker_tracks" \
  "${DATA_WAN}/alltracker_dense_tracks"

echo "─────────────────────────────────────────────"
echo "build_data_wan.sh"
echo "  DROID_ROOT      = ${DROID_ROOT}"
echo "  MODEL_PATH      = ${MODEL_PATH}"
echo "  DATA_WAN        = ${DATA_WAN}"
echo "  CAMERA_KEY      = ${CAMERA_KEY}    (serial = ${DROID_SERIAL})"
echo "  MAX_SCENES      = ${MAX_SCENES}    (0 = all)"
echo "  ALLTRACKER_ROOT = ${ALLTRACKER_ROOT}"
echo "  PY              = ${PY}"
echo "─────────────────────────────────────────────"

# ─── Phase 1: render DrRobot clips ───────────────────────────────────────────
if [[ "${SKIP_RENDER}" != "1" ]]; then
  echo ""
  echo "[1/3] Render DrRobot clips -> ${DATA_WAN}/clips/"
  RENDER_ARGS=(
    --model_path "${MODEL_PATH}"
    --dataset_root "${DROID_ROOT}"
    --output_dir "${DATA_WAN}/clips"
    --camera_key "${CAMERA_KEY}"
    --background "${RENDER_BG}"
    --fps "${RENDER_FPS}"
    --trajectory_camera
  )
  if [[ "${MAX_SCENES}" -gt 0 ]]; then
    RENDER_ARGS+=( --max_scenes "${MAX_SCENES}" )
  fi
  ( cd "${REPO_ROOT}/drrobot" && "${PY}" render_droid_scenes.py "${RENDER_ARGS[@]}" )
else
  echo "[1/3] SKIP_RENDER=1, skipping DrRobot rendering"
fi

# ─── Phase 2: AllTracker dense (+ sparse) tracks on real DROID camera ────────
if [[ "${SKIP_TRACKS}" != "1" ]]; then
  echo ""
  echo "[2/3] AllTracker dense + sparse tracks -> ${DATA_WAN}/alltracker_*"
  TRACKS_ARGS=(
    --alltracker_root "${ALLTRACKER_ROOT}"
    --droid_root "${DROID_ROOT}"
    --droid_camera_key "${CAMERA_KEY}"
    --out_tracks_dir "${DATA_WAN}/alltracker_tracks"
    --out_dense_tracks_dir "${DATA_WAN}/alltracker_dense_tracks"
    --out_viz_dir "${DATA_WAN}/alltracker_viz"
    --save_tracks
    --save_dense_tracks
    --image_size "${TRACK_IMAGE_SIZE}"
    --track_grid_x "${TRACK_GRID_X}"
    --track_grid_y "${TRACK_GRID_Y}"
    --no_viz
  )
  if [[ "${MAX_SCENES}" -gt 0 ]]; then
    TRACKS_ARGS+=( --max_scenes "${MAX_SCENES}" )
  fi
  "${PY}" "${REPO_ROOT}/scripts/extract_alltracker.py" "${TRACKS_ARGS[@]}"
else
  echo "[2/3] SKIP_TRACKS=1, skipping AllTracker extraction"
fi

# ─── Phase 3: symlink real videos + write metadata.csv ───────────────────────
if [[ "${SKIP_METADATA}" != "1" ]]; then
  echo ""
  echo "[3/3] Symlink real DROID RGB + write metadata.csv"
  PREP_ARGS=(
    --camera_key "${CAMERA_KEY}"
    --droid_root "${DROID_ROOT}"
    --data_wan "${DATA_WAN}"
  )
  if [[ "${MAX_SCENES}" -gt 0 ]]; then
    PREP_ARGS+=( --max_scenes "${MAX_SCENES}" )
  fi
  if [[ "${SKIP_RENDER}" == "1" ]]; then
    PREP_ARGS+=( --allow_missing_renders )
  fi
  "${PY}" "${REPO_ROOT}/prepare_data_wan.py" "${PREP_ARGS[@]}"
else
  echo "[3/3] SKIP_METADATA=1, skipping metadata.csv build"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
n_clips=$(find "${DATA_WAN}/clips" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
n_videos=$(find "${DATA_WAN}/videos" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
n_dense=$(find "${DATA_WAN}/alltracker_dense_tracks" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
n_sparse=$(find "${DATA_WAN}/alltracker_tracks" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)

echo ""
echo "─────────────────────────────────────────────"
echo "[done] data_wan layout @ ${DATA_WAN}"
echo "  clips/                   ${n_clips} mp4   (DrRobot action conditioning)"
echo "  videos/                  ${n_videos} mp4   (real DROID RGB target)"
echo "  alltracker_dense_tracks/ ${n_dense} npz   (dense per-pixel tracks)"
echo "  alltracker_tracks/       ${n_sparse} npz   (sparse 640-pt tracks)"
echo "  metadata.csv             $([[ -f "${DATA_WAN}/metadata.csv" ]] && echo present || echo MISSING)"
echo "─────────────────────────────────────────────"
