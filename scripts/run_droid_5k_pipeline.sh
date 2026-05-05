#!/usr/bin/env bash
#
# End-to-end wrapper: sample diverse DROID episodes, download them, and
# (optionally) extract npy artefacts and run the render + tracks pipeline.
#
# All phases are idempotent. Re-running resumes any phase that was interrupted.
#
# Usage:
#   ./scripts/run_droid_5k_pipeline.sh             # defaults: n=5000, ext1
#   N_EPISODES=1000 ./scripts/run_droid_5k_pipeline.sh
#   RUN_EXTRACT=1 RUN_RENDER=1 ./scripts/run_droid_5k_pipeline.sh
#
# Env vars (all optional, with defaults shown):
#   N_EPISODES=5000              target number of episodes
#   OUT_DIR=data/droid_5k        where scene_<idx>/ directories land
#   CACHE_DIR=data/droid_meta_cache   annotations cache (12 MB)
#   WORKERS=64                   thread pool size for HEAD + download
#   CAMERAS=ext1                 comma-separated: ext1,ext2,wrist
#   SEED=42                      stratified-sample RNG seed
#   HEAD_CHECK_N=0               0 = validate all candidates; >0 = subsample
#   RUN_EXTRACT=0                1 -> also run `extract` (npy + prompt.txt)
#   RUN_RENDER=0                 1 -> also run scale_data_generation.py
#   EXTRACT_PYTHON=...           python with h5py installed
#                                (default: dr conda env)
#   RENDER_OUT_DIR=data_wan_5k   render+tracks output (only if RUN_RENDER=1)
#   RENDER_GPUS="0 1 2 3"        GPUs for scale_data_generation
#

set -euo pipefail

# ─── Resolve repo-root paths ──────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOADER="${SCRIPT_DIR}/download_droid_5k.py"
RENDER_SCRIPT="${SCRIPT_DIR}/scale_data_generation.py"

# ─── Defaults ─────────────────────────────────────────────────────────────────

N_EPISODES="${N_EPISODES:-5000}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/droid_5k}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data/droid_meta_cache}"
WORKERS="${WORKERS:-64}"
CAMERAS="${CAMERAS:-ext1}"
SEED="${SEED:-42}"
HEAD_CHECK_N="${HEAD_CHECK_N:-0}"

RUN_EXTRACT="${RUN_EXTRACT:-0}"
RUN_RENDER="${RUN_RENDER:-0}"
EXTRACT_PYTHON="${EXTRACT_PYTHON:-/home/yjangir/miniconda3/envs/dr/bin/python}"
RENDER_OUT_DIR="${RENDER_OUT_DIR:-${REPO_ROOT}/data_wan_5k}"
RENDER_GPUS="${RENDER_GPUS:-0 1 2 3}"

PY="${PY:-python3}"

# ─── Sanity check ─────────────────────────────────────────────────────────────

if [[ ! -f "${DOWNLOADER}" ]]; then
    echo "FATAL: missing ${DOWNLOADER}" >&2
    exit 1
fi

# ─── Echo config ──────────────────────────────────────────────────────────────

echo "──────────────────────────────────────────────────────────────"
echo "DROID 5k pipeline"
echo "  N_EPISODES   = ${N_EPISODES}"
echo "  OUT_DIR      = ${OUT_DIR}"
echo "  CACHE_DIR    = ${CACHE_DIR}"
echo "  WORKERS      = ${WORKERS}"
echo "  CAMERAS      = ${CAMERAS}"
echo "  SEED         = ${SEED}"
echo "  HEAD_CHECK_N = ${HEAD_CHECK_N}"
echo "  RUN_EXTRACT  = ${RUN_EXTRACT}  (extract joints/intrinsics/extrinsics .npy)"
echo "  RUN_RENDER   = ${RUN_RENDER}   (run scale_data_generation.py)"
echo "──────────────────────────────────────────────────────────────"

mkdir -p "${OUT_DIR}" "${CACHE_DIR}"

# ─── Phase 1: fetch-meta ──────────────────────────────────────────────────────

echo
echo "[1/5] fetch-meta — annotations cache"
"${PY}" "${DOWNLOADER}" fetch-meta --cache_dir "${CACHE_DIR}"

# ─── Phase 2: sample ──────────────────────────────────────────────────────────

echo
echo "[2/5] sample — diverse stratified selection"
sample_args=(
    --cache_dir "${CACHE_DIR}"
    --out_dir   "${OUT_DIR}"
    --n         "${N_EPISODES}"
    --workers   "${WORKERS}"
    --seed      "${SEED}"
)
if [[ "${HEAD_CHECK_N}" -gt 0 ]]; then
    sample_args+=(--head_check_n "${HEAD_CHECK_N}")
fi
"${PY}" "${DOWNLOADER}" sample "${sample_args[@]}"

manifest="${OUT_DIR}/manifest.csv"
if [[ ! -s "${manifest}" ]]; then
    echo "FATAL: sample produced no manifest at ${manifest}" >&2
    exit 1
fi

# ─── Phase 3: download ────────────────────────────────────────────────────────

echo
echo "[3/5] download — parallel HTTPS pull"
"${PY}" "${DOWNLOADER}" download \
    --manifest "${manifest}" \
    --out_dir  "${OUT_DIR}" \
    --workers  "${WORKERS}" \
    --cameras  "${CAMERAS}"

n_scenes="$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'scene_*' | wc -l)"
echo "  -> ${n_scenes} scene directories present"

# ─── Phase 4: extract (optional) ──────────────────────────────────────────────

if [[ "${RUN_EXTRACT}" == "1" ]]; then
    echo
    echo "[4/5] extract — joints/intrinsics/extrinsics .npy + prompt.txt"
    if [[ ! -x "${EXTRACT_PYTHON}" ]]; then
        echo "  WARNING: ${EXTRACT_PYTHON} not executable; skipping extract." >&2
    else
        primary_cam="$(echo "${CAMERAS}" | cut -d, -f1)"
        "${EXTRACT_PYTHON}" "${DOWNLOADER}" extract \
            --out_dir     "${OUT_DIR}" \
            --camera_role "${primary_cam}"
    fi
else
    echo
    echo "[4/5] extract — skipped (RUN_EXTRACT=1 to enable)"
fi

# ─── Phase 5: render + tracks (optional) ──────────────────────────────────────

if [[ "${RUN_RENDER}" == "1" ]]; then
    echo
    echo "[5/5] render + tracks — scale_data_generation.py"
    if [[ ! -f "${RENDER_SCRIPT}" ]]; then
        echo "  FATAL: missing ${RENDER_SCRIPT}" >&2
        exit 1
    fi
    export MUJOCO_GL=egl
    "${PY}" "${RENDER_SCRIPT}" \
        --data_root "${OUT_DIR}" \
        --out_dir   "${RENDER_OUT_DIR}" \
        --gpus      ${RENDER_GPUS} \
        --no_viz
    echo "  -> renders + tracks under ${RENDER_OUT_DIR}"
else
    echo
    echo "[5/5] render+tracks — skipped (RUN_RENDER=1 to enable)"
    echo "      Manual:"
    echo "        export MUJOCO_GL=egl"
    echo "        python ${RENDER_SCRIPT} \\"
    echo "            --data_root ${OUT_DIR} \\"
    echo "            --out_dir   ${RENDER_OUT_DIR} \\"
    echo "            --gpus ${RENDER_GPUS} --no_viz"
fi

echo
echo "──────────────────────────────────────────────────────────────"
echo "Done."
echo "  manifest          : ${manifest}"
echo "  scenes downloaded : ${OUT_DIR}/scene_*/"
if [[ "${RUN_RENDER}" == "1" ]]; then
    echo "  render+tracks     : ${RENDER_OUT_DIR}/"
fi
echo "──────────────────────────────────────────────────────────────"
