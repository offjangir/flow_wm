#!/usr/bin/env bash
# Generate videos + metrics.json for legacy v1 and legacy EgoWM checkpoints at
# epochs 19 and 24, on train + held-out CSVs (same layout as epoch_29 runs).
#
# Usage:
#   ./scripts/eval_legacy_epochs_19_24.sh
#   CUDA_VISIBLE_DEVICES=0 ./scripts/eval_legacy_epochs_19_24.sh
#   PYTHON=/path/to/python ./scripts/eval_legacy_epochs_19_24.sh
#
# Two GPUs (v1 on physical GPU 0, EgoWM on physical GPU 1); each sees cuda:0:
#   PARALLEL_GPUS=1 ./scripts/eval_legacy_epochs_19_24.sh

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PYTHON:-/home/yjangir1/scratchhbharad2/users/yjangir1/conda-envs/dr/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-50}"
CFG="${CFG:-1.5}"
NS="${NS:-10}"

run_one() {
  local config="$1" ckpt="$2" csv="$3" out_dir="$4"
  echo "========== $out_dir =========="
  "$PY" scripts/generate_videos.py \
    --config "$config" \
    --ckpt "$ckpt" \
    --eval_csv "$csv" \
    --out_dir "$out_dir" \
    --num_inference_steps "$STEPS" \
    --cfg_scale "$CFG" \
    --num_samples "$NS" \
    --device "$DEVICE"
}

run_v1_epochs() {
  for ep in 19 24; do
    run_one \
      configs/train_drrobot_1k_legacy_8xl40.json \
      "checkpoints/wan_render_drrobot_1k_legacy_8xl40/run_20260506_233131/epoch_${ep}/render_conditioner.pt" \
      data_wan_1k/train_metadata_train.csv \
      "generations2/legacy_v1_run233131_TRAIN_epoch${ep}"

    run_one \
      configs/train_drrobot_1k_legacy_8xl40.json \
      "checkpoints/wan_render_drrobot_1k_legacy_8xl40/run_20260506_233131/epoch_${ep}/render_conditioner.pt" \
      data_wan_1k/train_metadata_test.csv \
      "generations2/legacy_v1_run233131_HELDOUT_epoch${ep}"
  done
}

run_egowm_epochs() {
  for ep in 19 24; do
    run_one \
      configs/train_drrobot_1k_legacy_8xl40_egowm.json \
      "checkpoints/wan_render_drrobot_1k_legacy_8xl40_egowm/run_20260507_052841/epoch_${ep}/render_conditioner.pt" \
      data_wan_1k/train_metadata_train.csv \
      "generations2/legacy_egowm_run052841_TRAIN_epoch${ep}"

    run_one \
      configs/train_drrobot_1k_legacy_8xl40_egowm.json \
      "checkpoints/wan_render_drrobot_1k_legacy_8xl40_egowm/run_20260507_052841/epoch_${ep}/render_conditioner.pt" \
      data_wan_1k/train_metadata_test.csv \
      "generations2/legacy_egowm_run052841_HELDOUT_epoch${ep}"
  done
}

if [[ "${PARALLEL_GPUS:-0}" == "1" ]]; then
  ( CUDA_VISIBLE_DEVICES="${V1_GPU:-0}" DEVICE=cuda:0 run_v1_epochs ) &
  pid1=$!
  ( CUDA_VISIBLE_DEVICES="${EGOWM_GPU:-1}" DEVICE=cuda:0 run_egowm_epochs ) &
  pid2=$!
  wait "$pid1" "$pid2"
else
  run_v1_epochs
  run_egowm_epochs
fi

echo "[done] all 8 eval runs finished."
