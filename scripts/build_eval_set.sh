#!/usr/bin/env bash
# Build a held-out 10-video evaluation set:
#   1. Sample 10 episodes from DROID NOT in data/droid_1k/manifest.csv
#   2. Download trajectory.h5 + ext1 MP4 for each
#   3. Render via DrRobot + extract AllTracker tracks
#   4. Build data_wan_eval/eval_metadata_10.csv (same schema as training metadata)
#
# Output:
#   data/droid_eval/scene_<id>/...        (raw DROID files)
#   data_wan_eval/renders/scene_<id>/drrobot_render.mp4
#   data_wan_eval/alltracker_tracks/scene_<id>.npz
#   data_wan_eval/clips/scene_<id>.mp4    (symlink to render)
#   data_wan_eval/videos/scene_<id>.mp4   (symlink to real DROID mp4)
#   data_wan_eval/eval_metadata_10.csv    (FINAL artefact for eval)
#
# Knobs:
#   N_EPS=10              sample size
#   SEED=99               sampling seed
#   GPUS="0 1 2 3"        GPUs for the render+tracks step (avoid GPUs in use)
#   PY=python3            interpreter (override: PY=/your/conda/bin/python ...)
#
set -euo pipefail

# Repo root = parent of this script's directory (no hardcoded WEKA/home paths).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Override if needed, e.g. PY=/path/to/conda-env/bin/python bash scripts/build_eval_set.sh
PY="${PY:-python3}"
N_EPS="${N_EPS:-10}"
SEED="${SEED:-99}"
GPUS="${GPUS:-0 1 2 3}"
LOG="$REPO/logs/build_eval_set.log"
mkdir -p "$REPO/logs"

cd "$REPO"

echo "=== $(date -Is) [1/5] Sample $N_EPS new episodes (excluding the trained 891) ===" | tee -a "$LOG"
"$PY" scripts/sample_eval_episodes.py \
    --cache_dir data/droid_meta_cache \
    --exclude_manifest data/droid_1k/manifest.csv \
    --n "$N_EPS" --seed "$SEED" \
    --out_manifest data/droid_eval/manifest.csv 2>&1 | tee -a "$LOG"

echo "=== $(date -Is) [2/5] Download $N_EPS episodes ===" | tee -a "$LOG"
"$PY" scripts/download_droid_5k.py download \
    --manifest data/droid_eval/manifest.csv \
    --out_dir  data/droid_eval \
    --workers 8 \
    --cameras ext1 2>&1 | tee -a "$LOG"

echo "=== $(date -Is) [3/5] DrRobot render + AllTracker tracks ===" | tee -a "$LOG"
MUJOCO_GL=egl PYTHONUNBUFFERED=1 "$PY" scripts/scale_data_generation.py \
    --data_root  data/droid_eval \
    --out_dir    data_wan_eval \
    --drrobot_config "$REPO/configs/drrobot_models.json" \
    --drrobot_root   "$REPO/drrobot" \
    --alltracker_root "$REPO/alltracker" \
    --gpus $GPUS \
    --no_viz --save_dense_tracks 2>&1 | tee -a "$LOG"

echo "=== $(date -Is) [4/5] Symlink renders -> clips/ ===" | tee -a "$LOG"
mkdir -p data_wan_eval/clips
n=0
for d in data_wan_eval/renders/scene_*; do
  scene=$(basename "$d")
  src="$d/drrobot_render.mp4"
  if [ -f "$src" ]; then
    ln -sfn "$(realpath "$src")" "data_wan_eval/clips/$scene.mp4"
    n=$((n+1))
  fi
done
echo "linked $n clip(s)" | tee -a "$LOG"

echo "=== $(date -Is) [5/5] Build eval_metadata via prepare_data_wan ===" | tee -a "$LOG"
"$PY" prepare_data_wan.py \
    --droid_root data/droid_eval \
    --data_wan   data_wan_eval \
    --camera_key ext1 2>&1 | tee -a "$LOG"

# prepare_data_wan writes "metadata.csv" — rename to make purpose explicit
mv data_wan_eval/metadata.csv data_wan_eval/eval_metadata.csv
ln -sfn "$(realpath data_wan_eval/eval_metadata.csv)" "$REPO/data_wan_1k/eval_metadata_held_out_10.csv"

echo "=== $(date -Is) DONE ===" | tee -a "$LOG"
echo "  eval CSV: data_wan_eval/eval_metadata.csv"
echo "  symlink:  data_wan_1k/eval_metadata_held_out_10.csv  (for eval_render_conditioning.py)"
echo "  eval hint: pass --dataset_base_path data_wan_eval (paths are under data_wan_eval/, not data_wan_1k/)"
ls -lh data_wan_eval/eval_metadata.csv
wc -l data_wan_eval/eval_metadata.csv
