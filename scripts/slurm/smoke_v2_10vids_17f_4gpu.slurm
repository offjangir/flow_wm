#!/bin/bash
#SBATCH --job-name=v2_smoke_4gpu
#SBATCH --account=becx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/work/nvme/becx/yjangir/flow_wm/logs/v2_smoke_4gpu_%j.out
#SBATCH --error=/work/nvme/becx/yjangir/flow_wm/logs/v2_smoke_4gpu_%j.err

source /work/nvme/becx/yjangir/miniconda3/etc/profile.d/conda.sh
conda activate dr

WM_ROOT=/work/nvme/becx/yjangir/flow_wm
cd "$WM_ROOT"
mkdir -p logs

export PYTHONPATH="$WM_ROOT/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[launch] job=$SLURM_JOB_ID host=$(hostname) gpus=4"
echo "[launch] v2 smoke: 10 videos × 3 epochs × 17 frames, no-zero-init out_proj"
echo "[launch] watch wandb for: train/action_adapter_out_proj_norm growth + val/loss_flow_mean"

accelerate launch \
  --num_processes 4 \
  --config_file "$WM_ROOT/configs/accelerate_fsdp.yaml" \
  -m world_model.wan_flow.train_fsdp \
  --config "$WM_ROOT/configs/train_drrobot_1k_10vids_v2_fsdp.json"
