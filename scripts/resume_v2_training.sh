#!/usr/bin/env bash
#
# Resume the v2 81-frame action-conditioned FSDP run from the most recent
# saved checkpoint. Designed for SLURM-style "walltime expires, get new
# allocation" workflow.
#
# What it does:
#   1. Auto-detects the latest run_*/epoch_N/render_conditioner.pt under the
#      configured output_dir.
#   2. Computes the correct --epoch_offset (= last_saved_epoch + 1) so new
#      epoch_* directories don't clobber the previous run's checkpoints.
#   3. Launches accelerate with --resume_render_ckpt + --epoch_offset.
#
# IMPORTANT: This is a *weight* resume only. AdamW optimizer state, LR
# scheduler step counter, and RNG state are NOT persisted across runs. You
# lose at most ``save_every_n_epochs`` epochs of optimizer momentum since the
# previous save (with default 5, that's the worst case). The model weights
# carry over.
#
# Usage:
#   ./scripts/resume_v2_training.sh
#
# Environment overrides:
#   CONFIG=...                  training config JSON (default: v2 full L40)
#   OUTPUT_DIR=...              run dir to scan for checkpoints (default: from CONFIG)
#   CUDA_VISIBLE_DEVICES=...    GPUs (default: 1,3,4,5,6,7 — skip GPU 0 + bad GPU 2)
#   NUM_GPUS=...                num_processes (default: count of CUDA_VISIBLE_DEVICES)
#   ACCEL_CONFIG=...            accelerate config (default: configs/accelerate_fsdp.yaml)
#   FROM_RUN=...                force resume from a specific run dir (basename); default = latest
#   FROM_EPOCH=...              force resume from a specific epoch index; default = latest in the run

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONFIG="${CONFIG:-configs/train_drrobot_1k_legacy_8xl40_action_aware_v2.json}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_fsdp.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')}"

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG"; exit 1; }
[[ -f "$ACCEL_CONFIG" ]] || { echo "ERROR: accelerate config not found: $ACCEL_CONFIG"; exit 1; }

# Read output_dir from the JSON config (no jq dependency — small awk parse).
OUTPUT_DIR="${OUTPUT_DIR:-$(python -c "
import json, sys
with open('$CONFIG') as f:
    cfg = json.load(f)
print(cfg['output_dir'].rstrip('/'))
")}"

[[ -d "$OUTPUT_DIR" ]] || { echo "ERROR: output_dir not found: $OUTPUT_DIR (no prior run to resume)"; exit 1; }

# ── Locate the run dir ─────────────────────────────────────────────────────
if [[ -n "${FROM_RUN:-}" ]]; then
    RUN_DIR="$OUTPUT_DIR/$FROM_RUN"
else
    # Newest run_* dir by mtime
    RUN_DIR=$(ls -td "$OUTPUT_DIR"/run_* 2>/dev/null | head -1 || true)
fi
[[ -d "${RUN_DIR:-}" ]] || { echo "ERROR: no run_* dir under $OUTPUT_DIR"; exit 1; }

# ── Locate the latest epoch_*/render_conditioner.pt ────────────────────────
if [[ -n "${FROM_EPOCH:-}" ]]; then
    LATEST_EPOCH="$FROM_EPOCH"
else
    # Pick the highest numeric epoch with a saved render_conditioner.pt
    LATEST_EPOCH=$(
        ls -d "$RUN_DIR"/epoch_* 2>/dev/null \
        | sed -n 's|.*/epoch_\([0-9]\+\)$|\1|p' \
        | sort -n \
        | while read e; do
            if [[ -f "$RUN_DIR/epoch_$e/render_conditioner.pt" ]]; then echo "$e"; fi
          done \
        | tail -1
    )
fi
[[ -n "${LATEST_EPOCH:-}" ]] || { echo "ERROR: no epoch_*/render_conditioner.pt under $RUN_DIR"; exit 1; }

CKPT="$RUN_DIR/epoch_${LATEST_EPOCH}/render_conditioner.pt"
EPOCH_OFFSET=$((LATEST_EPOCH + 1))

echo "================================================================="
echo " Resume v2 81f action-aware training"
echo "================================================================="
echo " repo            : $REPO"
echo " CONFIG          : $CONFIG"
echo " ACCEL_CONFIG    : $ACCEL_CONFIG"
echo " GPUs            : $CUDA_VISIBLE_DEVICES  (NUM_GPUS=$NUM_GPUS)"
echo " OUTPUT_DIR      : $OUTPUT_DIR"
echo " RUN_DIR         : $RUN_DIR"
echo " latest epoch    : $LATEST_EPOCH"
echo " resume ckpt     : $CKPT"
echo " --epoch_offset  : $EPOCH_OFFSET"
echo "================================================================="
echo

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate dr 2>/dev/null || true
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES

LOG="logs/train_v2_81f_resume_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
echo "[resume] log file: $LOG"

# Detach via setsid + nohup so the run survives the launching shell.
setsid nohup accelerate launch \
    --main_process_port 0 \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$NUM_GPUS" \
    -m world_model.wan_flow.train_fsdp \
    --config "$CONFIG" \
    --resume_render_ckpt "$CKPT" \
    --epoch_offset "$EPOCH_OFFSET" \
    > "$LOG" 2>&1 < /dev/null &
disown
PID=$!

echo "[resume] launched PID=$PID  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[resume] tail with:"
echo "  tail -f $LOG"
