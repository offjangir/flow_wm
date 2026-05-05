#!/bin/bash
# setup_dr_env_and_checkpoints.sh
# This script ensures the 'dr' environment is set up and pulls necessary checkpoints.

set -e

# Find conda base and source it to make 'conda activate' work inside the script
CONDA_PATH=$(conda info --base)
source "${CONDA_PATH}/etc/profile.d/conda.sh"

ENV_NAME="dr"

echo "========================================="
echo " Step 1: Verifying Conda Environment"
echo "========================================="

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists. Activating..."
    conda activate "${ENV_NAME}"
else
    echo "Conda environment '${ENV_NAME}' not found."
    if [ -f "env.yaml" ]; then
        echo "Building environment from env.yaml..."
        conda env create -f env.yaml -y
        conda activate "${ENV_NAME}"
    else
        echo "env.yaml not found! Setting up manual fallback..."
        conda create -n "${ENV_NAME}" python=3.10 -y
        conda activate "${ENV_NAME}"
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    fi
fi

echo "========================================="
echo " Step 2: Installing/Verifying Dependencies"
echo "========================================="

# Ensure latest huggingface_hub is installed for downloads
pip install --upgrade huggingface_hub

if [ -f "requirements-droid-drrobot-any4d.txt" ]; then
    echo "Installing from requirements file..."
    pip install -r requirements-droid-drrobot-any4d.txt
fi

echo "Installing current package as editable..."
pip install -e .

# Extra crucial dependencies from our previous troubleshooting
pip install scikit-image einops imageio-ffmpeg ftfy

echo "========================================="
echo " Step 3: Pulling Model Checkpoints"
echo "========================================="

# Pull the base Wan 2.1 I2V 14B Diffusers weights
python -c "
from huggingface_hub import snapshot_download
import sys

repo_id = 'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers'
print(f'Checking/Downloading weights for {repo_id}...')

try:
    path = snapshot_download(repo_id=repo_id)
    print(f'Successfully pulled checkpoint to: {path}')
except Exception as e:
    print(f'ERROR pulling checkpoints: {e}', file=sys.stderr)
    sys.exit(1)
"

echo "========================================="
echo " All systems ready for 'dr' environment!"
echo "========================================="
