#!/bin/bash
#SBATCH --job-name=wan-fsdp-8
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:8
#SBATCH --cpus-per-task=112
#SBATCH --mem=512G
#SBATCH --time=48:00:00
# Absolute paths so logs always land under the repo (independent of `sbatch` cwd).
# Same path for --output and --error merges stdout + stderr into one file.
#SBATCH --output=/home/yjangir1/scratchhbharad2/users/yjangir1/flow_wm/logs/%x-%j.log
#SBATCH --error=/home/yjangir1/scratchhbharad2/users/yjangir1/flow_wm/logs/%x-%j.log
#
# 8×GPU FSDP training (Wan `train_fsdp.py`) with conda env `dr`.
#
# Submit from repo root (recommended):
#   cd /home/yjangir1/scratchhbharad2/users/yjangir1/flow_wm && sbatch scripts/slurm/train_wan_fsdp_8gpu.slurm
#
# Override training config (optional):
#   sbatch --export=ALL,TRAIN_CONFIG=configs/train_drrobot_fsdp.json scripts/slurm/train_wan_fsdp_8gpu.slurm
#
# Notes:
# - `--cpus-per-task` = 14 × 8 GPUs (tune to your cluster’s --cpus-per-gpu policy).
# - Raise `--mem` if you still see exit -9 (OOM) with FSDP CPU offload.
# - Change `#SBATCH --partition` / `--gres` if you use h100 / l40s (e.g. gpu:h100:8).
#
# Logs: Slurm writes this script’s stdout/stderr (and Python print/tqdm) to:
#   ${REPO_ROOT}/logs/<jobname>-<jobid>.log
# Optional duplicate copy (same content) for a stable path to `tail -f`:
#   sbatch --export=ALL,ALSO_LOG=/path/to/my-run.log  (see below)

set -euo pipefail

# Repo root: directory that contains `src/` and `configs/`
REPO_ROOT="${REPO_ROOT:-/home/yjangir1/scratchhbharad2/users/yjangir1/flow_wm}"
cd "${REPO_ROOT}"
mkdir -p logs

# Stream everything through `tee` so an extra log file gets a copy (optional).
# Default: only Slurm’s merged log (SBATCH --output/--error above).
if [[ -n "${ALSO_LOG:-}" ]]; then
	mkdir -p "$(dirname "${ALSO_LOG}")"
	exec > >(tee -a "${ALSO_LOG}") 2>&1
fi

# Flush Python prints / tqdm to the log without waiting for buffer full
export PYTHONUNBUFFERED=1

echo "[slurm] merged Slurm log (stdout+stderr): ${REPO_ROOT}/logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-?}.log"
[[ -n "${ALSO_LOG:-}" ]] && echo "[slurm] duplicate stream tee → ${ALSO_LOG}"

# ---- Conda (scratch install; works in non-interactive Slurm shells) ----
CONDA_ROOT="${CONDA_ROOT:-/home/yjangir1/scratchhbharad2/users/yjangir1/miniconda3}"
CONDA_ENV_DR="${CONDA_ENV_DR:-/home/yjangir1/scratchhbharad2/users/yjangir1/conda-envs/dr}"
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
	echo "ERROR: conda.sh not found at ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
	exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_DR}"

# ---- Python path (package lives under src/world_model) ----
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# ---- Hugging Face cache on scratch (override if your site differs) ----
export HF_HOME="${HF_HOME:-/home/yjangir1/scratchhbharad2/users/yjangir1}"

# ---- Distributed (Slurm may set some of these; do not hard-fail) ----
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_vanilla_i2v_fsdp.json}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_fsdp.yaml}"

echo "[slurm] host=$(hostname) date=$(date -Is)"
echo "[slurm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[slurm] SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "[slurm] REPO_ROOT=${REPO_ROOT} TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "[slurm] python=$(command -v python) $(python -V 2>&1)"

# 8 processes on this node; must match allocated GPUs.
exec accelerate launch \
	--num_processes 8 \
	--num_machines 1 \
	--machine_rank 0 \
	--main_process_port "${MAIN_PROCESS_PORT:-0}" \
	--config_file "${ACCEL_CONFIG}" \
	"${REPO_ROOT}/src/world_model/wan_flow/train_fsdp.py" \
	--config "${TRAIN_CONFIG}"
