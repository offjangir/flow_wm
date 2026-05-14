#!/bin/bash
#SBATCH --job-name=wan_8blocks
#SBATCH --account=becx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/work/nvme/becx/yjangir/flow_wm/logs/8blocks_%j.out
#SBATCH --error=/work/nvme/becx/yjangir/flow_wm/logs/8blocks_%j.err

# ---- environment -------------------------------------------------------
source /work/nvme/becx/yjangir/miniconda3/etc/profile.d/conda.sh
conda activate dr

WM_ROOT=/work/nvme/becx/yjangir/flow_wm
cd "$WM_ROOT"
mkdir -p logs

export PYTHONPATH=/work/nvme/becx/yjangir/flow_wm/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[launch] job=$SLURM_JOB_ID  nodes=1"
echo "[launch] config: train_8blocks_8gpu.json"

# ---- launch -----------------------------------------------------------
accelerate launch \
  --config_file /work/nvme/becx/yjangir/flow_wm/configs/accelerate_fsdp.yaml \
  --num_processes 4 \
  /work/nvme/becx/yjangir/flow_wm/src/world_model/wan_flow/train_fsdp.py \
  --config /work/nvme/becx/yjangir/flow_wm/configs/train_8blocks_8gpu.json

