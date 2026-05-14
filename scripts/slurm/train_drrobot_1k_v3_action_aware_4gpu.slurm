#!/bin/bash
#SBATCH --job-name=v3_aa_4gpu
#SBATCH --account=becx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --output=/work/nvme/becx/yjangir/flow_wm/logs/v3_aa_4gpu_%j.out
#SBATCH --error=/work/nvme/becx/yjangir/flow_wm/logs/v3_aa_4gpu_%j.err

source /work/nvme/becx/yjangir/miniconda3/etc/profile.d/conda.sh
conda activate dr

WM_ROOT=/work/nvme/becx/yjangir/flow_wm
cd "$WM_ROOT"
mkdir -p logs

export PYTHONPATH="$WM_ROOT/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[launch] job=$SLURM_JOB_ID host=$(hostname) gpus=4"
echo "[launch] v3 full run: 802 videos x 30 epochs x 17 frames"
echo "[launch] ActionRenderCrossAdapterV3 (residual transformer blocks, standard init)"
echo "[launch] A/B target: egowm 30-epoch baseline run_20260507_052841 (~16.2 dB)"
echo "[launch] smoke validated: 10-vid x 10-epoch val loss 1.52 -> 0.33 monotonic"

accelerate launch \
  --num_processes 4 \
  --config_file "$WM_ROOT/configs/accelerate_fsdp.yaml" \
  -m world_model.wan_flow.train_fsdp \
  --config "$WM_ROOT/configs/train_drrobot_1k_legacy_8xl40_action_aware_v3.json"
