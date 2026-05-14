#!/bin/bash
#SBATCH --account=becx-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

source /work/nvme/becx/yjangir/miniconda3/etc/profile.d/conda.sh
conda activate dr
export PYTHONPATH=/work/nvme/becx/yjangir/flow_wm/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/eval_world_model.py \
    --model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --checkpoint_path /work/nvme/becx/yjangir/flow_wm/checkpoints/wan_8blocks_8gpu/run_20260429_035114/epoch_19/render_conditioner.pt \
    --metadata_csv /work/nvme/becx/yjangir/flow_wm/data_wan/metadata.csv \
    --dataset_base_path /work/nvme/becx/yjangir/flow_wm/data_wan \
    --output_dir /work/nvme/becx/yjangir/flow_wm/eval_outputs/run_20260429_035114_epoch_19 \
    --num_samples 3 \
    --num_frames 17 \
    --height 320 \
    --width 576 \
    --mixed_precision bf16 \
    --no-cpu_offload
