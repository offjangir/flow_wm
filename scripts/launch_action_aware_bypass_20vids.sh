#!/usr/bin/env bash
#
# Launch FSDP training of the render-conditioned Wan I2V world model with the
# action-aware AdaLN pathway in *bypass* mode (no actions; spatial render
# features only) on a 20-video subset of the dataset.
#
# What it does:
#   1. Builds data_wan_data/metadata_20vids.csv from the first 20 rows of
#      data_wan_data/metadata.csv (idempotent — skipped if file exists).
#   2. Launches FSDP training with the action-aware bypass config.
#
# Env vars (all optional):
#   NUM_GPUS=4              GPU count for accelerate launch
#   METADATA_SRC=...        source metadata CSV (default: data_wan_data/metadata.csv)
#   METADATA_DST=...        20-video subset path (default: data_wan_data/metadata_20vids.csv)
#   CONFIG=...              training config JSON (default: action-aware bypass)
#   ACCEL_CONFIG=...        accelerate config YAML (default: configs/accelerate_fsdp.yaml)
#   FORCE_REBUILD=1         rebuild the 20-video CSV even if it exists
#
# Usage:
#   ./scripts/launch_action_aware_bypass_20vids.sh
#   NUM_GPUS=8 ./scripts/launch_action_aware_bypass_20vids.sh

set -euo pipefail

# ─── repo root ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ─── defaults ───────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-4}"
METADATA_SRC="${METADATA_SRC:-data_wan_data/metadata.csv}"
METADATA_DST="${METADATA_DST:-data_wan_data/metadata_20vids.csv}"
CONFIG="${CONFIG:-configs/train_drrobot_action_aware_bypass_20vids.json}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_fsdp.yaml}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

# ─── sanity checks ──────────────────────────────────────────────────────────
[[ -f "$METADATA_SRC" ]] || { echo "ERROR: source metadata not found: $METADATA_SRC"; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: training config not found: $CONFIG"; exit 1; }
[[ -f "$ACCEL_CONFIG" ]] || { echo "ERROR: accelerate config not found: $ACCEL_CONFIG"; exit 1; }

# ─── 1. build the 20-video metadata subset ──────────────────────────────────
if [[ "$FORCE_REBUILD" == "1" || ! -f "$METADATA_DST" ]]; then
    echo "[1/2] Building 20-video subset:  $METADATA_DST"
    head -1 "$METADATA_SRC" > "$METADATA_DST"
    tail -n +2 "$METADATA_SRC" | head -20 >> "$METADATA_DST"
    n_rows=$(($(wc -l < "$METADATA_DST") - 1))
    echo "      wrote $n_rows data rows (header + $n_rows = $((n_rows + 1)) lines)"
    if [[ "$n_rows" -lt 20 ]]; then
        echo "      WARNING: source CSV had fewer than 20 data rows ($n_rows). Continuing anyway."
    fi
else
    echo "[1/2] Reusing existing $METADATA_DST  (set FORCE_REBUILD=1 to overwrite)"
fi

# ─── 2. launch training ─────────────────────────────────────────────────────
echo "[2/2] Launching FSDP training on ${NUM_GPUS} GPU(s)"
echo "      config        : $CONFIG"
echo "      accelerate cfg: $ACCEL_CONFIG"
echo

exec accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$NUM_GPUS" \
    -m world_model.wan_flow.train_fsdp \
    --config "$CONFIG"
