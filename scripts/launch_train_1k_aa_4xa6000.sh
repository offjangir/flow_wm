#!/usr/bin/env bash
#
# Launch FSDP training of the render-conditioned + action-aware Wan I2V
# world model on the 882-clip droid_1k dataset, 4x A6000 GPUs.
#
# What it does:
#   1. Activates the `dr` conda env (where torch, diffusers, accelerate live).
#   2. Sanity-checks the metadata CSV and accelerate / training configs.
#   3. Launches FSDP training via `accelerate launch`.
#
# Env vars (all optional):
#   NUM_GPUS=4              GPU count for accelerate launch
#   CONFIG=...              training config JSON
#                           (default: configs/train_drrobot_1k_aa_4xa6000.json)
#   ACCEL_CONFIG=...        accelerate config YAML
#                           (default: configs/accelerate_fsdp.yaml)
#   CONDA_ENV=dr            conda env to activate
#
# Usage:
#   ./scripts/launch_train_1k_aa_4xa6000.sh
#   NUM_GPUS=2 ./scripts/launch_train_1k_aa_4xa6000.sh

set -euo pipefail

# ─── repo root ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ─── defaults ───────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-4}"
CONFIG="${CONFIG:-configs/train_drrobot_1k_aa_4xa6000.json}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_fsdp.yaml}"
CONDA_ENV="${CONDA_ENV:-dr}"

# ─── conda activate ─────────────────────────────────────────────────────────
# Source conda.sh so `conda activate` works in non-interactive shells.
CONDA_SH=""
for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$candidate" ]]; then
        CONDA_SH="$candidate"
        break
    fi
done
if [[ -z "$CONDA_SH" ]]; then
    echo "ERROR: cannot find conda.sh; set CONDA_SH manually if conda lives elsewhere." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
echo "[env] python=$(which python)  ($CONDA_ENV)"

# ─── sanity checks ──────────────────────────────────────────────────────────
[[ -f "$CONFIG" ]] || { echo "ERROR: training config not found: $CONFIG"; exit 1; }
[[ -f "$ACCEL_CONFIG" ]] || { echo "ERROR: accelerate config not found: $ACCEL_CONFIG"; exit 1; }

# Pull dataset paths from the JSON via python (json module — no jq dep).
# Pass the path via env so we don't have to interpolate into a heredoc.
read -r DATASET_BASE METADATA_CSV < <(CONFIG="$CONFIG" python - <<'PY'
import json, os
cfg = json.load(open(os.environ["CONFIG"]))
print(cfg["dataset_base_path"], cfg["metadata_csv"])
PY
)

[[ -d "$DATASET_BASE" ]] || { echo "ERROR: dataset_base_path does not exist: $DATASET_BASE"; exit 1; }
[[ -f "$METADATA_CSV" ]] || { echo "ERROR: metadata CSV does not exist: $METADATA_CSV"; exit 1; }
N_ROWS=$(($(wc -l < "$METADATA_CSV") - 1))
echo "[data] $METADATA_CSV  ($N_ROWS rows)"

# ─── launch ─────────────────────────────────────────────────────────────────
echo "[launch] FSDP training on ${NUM_GPUS} GPU(s)"
echo "         config        : $CONFIG"
echo "         accelerate cfg: $ACCEL_CONFIG"
echo

exec accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$NUM_GPUS" \
    -m world_model.wan_flow.train_fsdp \
    --config "$CONFIG"
