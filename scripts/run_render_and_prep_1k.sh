#!/usr/bin/env bash
# Full pipeline: DrRobot render + AllTracker tracks for all scenes in droid_1k,
# then build data_wan/metadata.csv for training.
#
# Run with nohup + disown so it survives the launching shell ending.
# Logs go to logs/render_prep_1k.log.
set -euo pipefail

REPO=/weka/scratch/hbharad2/users/yjangir1/flow_wm
PY=/home/yjangir1/scratchhbharad2/users/yjangir1/conda-envs/dr/bin/python
DATA_ROOT=$REPO/data/droid_1k
OUT_DIR=$REPO/data_wan_1k
LOG=$REPO/logs/render_prep_1k.log

mkdir -p "$OUT_DIR" "$REPO/logs"
echo "=== $(date -Is) Phase 2: DrRobot render + AllTracker tracks ===" | tee -a "$LOG"

cd "$REPO"
MUJOCO_GL=egl PYTHONUNBUFFERED=1 "$PY" scripts/scale_data_generation.py \
  --data_root "$DATA_ROOT" \
  --out_dir   "$OUT_DIR" \
  --drrobot_config "$REPO/configs/drrobot_models.json" \
  --drrobot_root   "$REPO/drrobot" \
  --alltracker_root "$REPO/alltracker" \
  --gpus ${RENDER_GPUS:-0 2 5 6} \
  --no_viz --save_dense_tracks \
  >> "$LOG" 2>&1

echo "=== $(date -Is) Phase 2 done ===" | tee -a "$LOG"

# Remux every render with +faststart in place (moov atom at the front so Cursor
# / browsers can preview without downloading the whole file). -c copy = no
# re-encode, just rewrites the container; ~milliseconds per file.
echo "=== $(date -Is) Remuxing renders with +faststart ===" | tee -a "$LOG"
FFMPEG=$("$PY" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
n_remux=0
for f in "$OUT_DIR"/renders/scene_*/drrobot_render.mp4; do
  [ -f "$f" ] || continue
  tmp="${f%.mp4}.faststart.mp4"
  "$FFMPEG" -y -loglevel error -i "$f" -c copy -movflags +faststart "$tmp" \
    && mv -f "$tmp" "$f" && n_remux=$((n_remux+1))
done
echo "remuxed $n_remux render(s)" | tee -a "$LOG"

# Mirror renders to data_wan_1k/clips/<scene>.mp4 so prepare_data_wan.py finds them.
echo "=== $(date -Is) Symlinking renders -> clips/ ===" | tee -a "$LOG"
mkdir -p "$OUT_DIR/clips"
n_clips=0
for d in "$OUT_DIR"/renders/scene_*; do
  scene=$(basename "$d")
  src="$d/drrobot_render.mp4"
  if [ -f "$src" ]; then
    ln -sfn "$src" "$OUT_DIR/clips/$scene.mp4"
    n_clips=$((n_clips+1))
  fi
done
echo "linked $n_clips clip(s)" | tee -a "$LOG"

echo "=== $(date -Is) Phase 3: prepare_data_wan.py ===" | tee -a "$LOG"
"$PY" prepare_data_wan.py \
  --droid_root "$DATA_ROOT" \
  --data_wan   "$OUT_DIR" \
  --camera_key ext1 \
  >> "$LOG" 2>&1

echo "=== $(date -Is) ALL DONE ===" | tee -a "$LOG"
