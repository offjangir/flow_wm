#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for wm:
# 1) create/update conda env "dr"
# 2) install dependencies
# 3) download required HF model checkpoints used by configs/*.json
#
# Usage:
#   bash scripts/setup_dr_env_and_checkpoints.sh
#
# Optional env vars:
#   ENV_NAME=dr
#   PYTHON_VERSION=3.10
#   HF_HOME=/path/to/hf/cache
#   HF_TOKEN=hf_xxx   # if model access requires auth

ENV_NAME="${ENV_NAME:-dr}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQ_FILE="${WM_ROOT}/requirements-droid-drrobot-any4d.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] ERROR: conda not found in PATH."
  echo "[setup] Install Miniconda/Anaconda first."
  exit 1
fi

# Ensure "conda activate" works in non-interactive shell.
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | rg -x "${ENV_NAME}" >/dev/null; then
  echo "[setup] Conda env '${ENV_NAME}' already exists."
else
  echo "[setup] Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

echo "[setup] Activating env '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

echo "[setup] Installing PyTorch CUDA stack..."
conda install -n "${ENV_NAME}" -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

echo "[setup] Installing pip tooling..."
python -m pip install --upgrade pip setuptools wheel

if [[ -f "${REQ_FILE}" ]]; then
  echo "[setup] Installing pipeline requirements from ${REQ_FILE}..."
  python -m pip install -r "${REQ_FILE}"
fi

echo "[setup] Installing wm package (editable)..."
python -m pip install -e "${WM_ROOT}"

if [[ -d "${WM_ROOT}/../Any4D" ]]; then
  echo "[setup] Installing Any4D editable (no deps)..."
  python -m pip install -e "${WM_ROOT}/../Any4D" --no-deps
else
  echo "[setup] NOTE: ${WM_ROOT}/../Any4D not found. Skipping editable Any4D install."
fi

echo "[setup] Installing small utility deps used by scripts..."
python -m pip install "scikit-image>=0.22" "einops>=0.8" imageio-ffmpeg "huggingface_hub>=0.24"

echo "[setup] Discovering required model checkpoints from configs/*.json..."
MODEL_IDS="$(
python - "${WM_ROOT}" <<'PY'
import glob
import json
import os
import sys

wm_root = sys.argv[1]
ids = set()
for p in glob.glob(os.path.join(wm_root, "configs", "*.json")):
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    mp = data.get("model_path")
    if isinstance(mp, str) and "/" in mp and not os.path.isdir(mp):
        ids.add(mp.strip())

# default used throughout repo if configs change
ids.add("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")

for m in sorted(ids):
    print(m)
PY
)"

if [[ -z "${MODEL_IDS}" ]]; then
  echo "[setup] No remote HF model IDs detected; skipping checkpoint download."
  exit 0
fi

HF_LOCAL_ROOT="${WM_ROOT}/checkpoints/hf_models"
mkdir -p "${HF_LOCAL_ROOT}"

echo "[setup] Downloading/checking HF model snapshots under ${HF_LOCAL_ROOT}..."
export MODEL_IDS
python - "${HF_LOCAL_ROOT}" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

hf_root = sys.argv[1]
token = os.environ.get("HF_TOKEN")
models = [m.strip() for m in os.environ.get("MODEL_IDS", "").splitlines() if m.strip()]

allow_patterns = [
    "transformer/*",
    "vae/*",
    "scheduler/*",
    "image_encoder/*",
    "image_processor/*",
    "tokenizer/*",
    "text_encoder/*",
    "*.json",
]

for model_id in models:
    local_dir = os.path.join(hf_root, model_id.replace("/", "__"))
    os.makedirs(local_dir, exist_ok=True)
    print(f"[setup] -> {model_id}")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            token=token,
            resume_download=True,
        )
    except Exception as e:
        print(f"[setup] WARNING: failed downloading {model_id}: {e}")
        print("[setup] If this is gated/private, set HF_TOKEN and rerun.")

print("[setup] HF checkpoint sync complete.")
PY

echo ""
echo "[setup] DONE."
echo "[setup] Env: ${ENV_NAME}"
echo "[setup] To use:"
echo "        conda activate ${ENV_NAME}"
echo "        cd ${WM_ROOT}"
