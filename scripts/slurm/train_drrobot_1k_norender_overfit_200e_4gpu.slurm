#!/bin/bash
#SBATCH --job-name=nordr_s200_4gpu
#SBATCH --account=becx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/work/nvme/becx/yjangir/flow_wm/logs/nordr_s200_4gpu_%j.out
#SBATCH --error=/work/nvme/becx/yjangir/flow_wm/logs/nordr_s200_4gpu_%j.err

source /work/nvme/becx/yjangir/miniconda3/etc/profile.d/conda.sh
conda activate dr

WM_ROOT=/work/nvme/becx/yjangir/flow_wm
cd "$WM_ROOT"
mkdir -p logs

export PYTHONPATH="$WM_ROOT/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[launch] job=$SLURM_JOB_ID host=$(hostname) gpus=4"
echo "[launch] config: configs/train_drrobot_1k_legacy_8xl40_egowm.json (base)"
echo "[launch] OVERRIDES: trainable=full_dit, drop_render_conditioning, lambda_tracks=0, num_epochs=200, save_every=100, separate output_dir"
echo "[launch] vanilla Wan 2.1 I2V fine-tune (no render adapter, no tracks); overfit-on-all-videos baseline"

accelerate launch \
  --num_processes 4 \
  --config_file "$WM_ROOT/configs/accelerate_fsdp.yaml" \
  -m world_model.wan_flow.train_fsdp \
  --config "$WM_ROOT/configs/train_drrobot_1k_legacy_8xl40_egowm.json" \
  --trainable full_dit \
  --drop_render_conditioning \
  --lambda_tracks 0.0 \
  --num_epochs 200 \
  --save_every_n_epochs 100 \
  --output_dir "$WM_ROOT/checkpoints/wan_render_drrobot_1k_norender_overfit" \
  --wandb_run_name 1k_norender_overfit_200e_4gpu
