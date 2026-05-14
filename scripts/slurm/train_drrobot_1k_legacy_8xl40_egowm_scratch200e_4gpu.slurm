#!/bin/bash
#SBATCH --job-name=egowm_s200_4gpu
#SBATCH --account=becx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/work/nvme/becx/yjangir/flow_wm/logs/egowm_s200_4gpu_%j.out
#SBATCH --error=/work/nvme/becx/yjangir/flow_wm/logs/egowm_s200_4gpu_%j.err

source /work/nvme/becx/yjangir/miniconda3/etc/profile.d/conda.sh
conda activate dr

WM_ROOT=/work/nvme/becx/yjangir/flow_wm
cd "$WM_ROOT"
mkdir -p logs

export PYTHONPATH="$WM_ROOT/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[launch] job=$SLURM_JOB_ID host=$(hostname) gpus=4"
echo "[launch] config: configs/train_drrobot_1k_legacy_8xl40_egowm.json"
echo "[launch] from scratch (no --resume_render_ckpt), 200 epochs, save every 25"

accelerate launch \
  --num_processes 4 \
  --config_file "$WM_ROOT/configs/accelerate_fsdp.yaml" \
  -m world_model.wan_flow.train_fsdp \
  --config "$WM_ROOT/configs/train_drrobot_1k_legacy_8xl40_egowm.json" \
  --num_epochs 200 \
  --save_every_n_epochs 25 \
  --wandb_run_name 1k_legacy_8xl40_egowm_scratch_200e_4gpu
