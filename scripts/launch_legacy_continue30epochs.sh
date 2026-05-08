#!/usr/bin/env bash
# Continue legacy v1 and legacy EgoWM 1k runs for 30 more epochs (epochs 30–59
# in logs and checkpoint dirs), loading epoch_29 render_conditioner.pt each.
#
# Produces NEW run_<timestamp>/ under each experiment output_dir (same as a
# fresh launch). Optimizer state is re-initialized (AdamW fresh); LR schedule
# runs over these 30 epochs only.
#
# Single node 8× GPU (matches prior full runs):
#   cd /path/to/flow_wm && ./scripts/launch_legacy_continue30epochs.sh
#
# Two nodes in parallel (e.g. v1 on GPU set 0–7, egowm on another machine): run
# only the first or second block below.
#
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

# Match GPU count to your node (prior full runs used 8× L40).
NGPU="${NGPU:-8}"
ACC="accelerate launch --num_processes ${NGPU} --config_file configs/accelerate_fsdp.yaml -m world_model.wan_flow.train_fsdp"

VARIANT="${VARIANT:-both}"  # v1 | egowm | both (both runs sequentially in this shell)

run_v1() {
  $ACC \
    --config configs/train_drrobot_1k_legacy_8xl40.json \
    --resume_render_ckpt checkpoints/wan_render_drrobot_1k_legacy_8xl40/run_20260506_233131/epoch_29/render_conditioner.pt \
    --epoch_offset 30 \
    --num_epochs 30 \
    --wandb_run_name 1k_legacy_8xl40_v1_continue30e
}

run_egowm() {
  $ACC \
    --config configs/train_drrobot_1k_legacy_8xl40_egowm.json \
    --resume_render_ckpt checkpoints/wan_render_drrobot_1k_legacy_8xl40_egowm/run_20260507_052841/epoch_29/render_conditioner.pt \
    --epoch_offset 30 \
    --num_epochs 30 \
    --wandb_run_name 1k_legacy_8xl40_egowm_continue30e
}

case "$VARIANT" in
  v1)    run_v1 ;;
  egowm) run_egowm ;;
  both)
    run_v1
    run_egowm
    echo "[done] both continue-30 jobs finished (sequential on this shell)."
    ;;
  *)
    echo "VARIANT must be v1, egowm, or both; got $VARIANT" >&2
    exit 1
    ;;
esac

if [[ "$VARIANT" != both ]]; then
  echo "[done] VARIANT=$VARIANT job finished."
fi
