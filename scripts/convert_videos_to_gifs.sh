#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFMPEG="${FFMPEG:-ffmpeg}"
FFPROBE="${FFPROBE:-ffprobe}"
LOG_DIR="${ROOT}/logs"
FORCE="${FORCE:-0}"
mkdir -p "$LOG_DIR"

activate_ffmpeg() {
  if command -v "$FFMPEG" >/dev/null 2>&1 && command -v "$FFPROBE" >/dev/null 2>&1; then
    return 0
  fi

  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate dr
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/opt/conda/etc/profile.d/conda.sh"
    conda activate dr
  fi

  FFMPEG=ffmpeg
  FFPROBE=ffprobe
  command -v "$FFMPEG" >/dev/null 2>&1
  command -v "$FFPROBE" >/dev/null 2>&1
}

target_fps() {
  local mp4="$1"
  local rate
  rate="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nw=1:nk=1 "$mp4")"
  python3 - "$rate" <<'PY'
import sys
from fractions import Fraction

rate = Fraction(sys.argv[1])
out = max(Fraction(1, 1), rate / 3)
if out.denominator == 1:
    print(out.numerator)
else:
    print(f"{out.numerator}/{out.denominator}")
PY
}

convert_tree() {
  local src_root="$1"
  local label="$2"
  local log_file="$LOG_DIR/convert_gifs_${label}_$(date +%Y%m%d_%H%M%S).log"
  local total done skipped failed

  mapfile -d '' -t mp4_files < <(find "$src_root" -type f -name '*.mp4' -print0 | sort -z)
  total="${#mp4_files[@]}"
  done=0
  skipped=0
  failed=0

  {
    echo "source_root=$src_root"
    echo "total_mp4=$total"
    echo "force=$FORCE"
    echo "started_at=$(date -Is)"
  } | tee "$log_file"

  for mp4 in "${mp4_files[@]}"; do
    gif="${mp4%.mp4}.gif"
    if [[ "$FORCE" != "1" && -f "$gif" && "$gif" -nt "$mp4" ]]; then
      skipped=$((skipped + 1))
      continue
    fi

    fps_out="$(target_fps "$mp4")"
    if "$FFMPEG" -y -hide_banner -loglevel error -i "$mp4" \
      -vf "setpts=3.0*PTS,fps=${fps_out},split[s0][s1];[s0]palettegen=stats_mode=diff:max_colors=256[p];[s1][p]paletteuse=dither=sierra2_4a" \
      -loop 0 "$gif"; then
      done=$((done + 1))
      echo "ok fps=${fps_out} $mp4"
    else
      failed=$((failed + 1))
      echo "fail fps=${fps_out} $mp4" >&2
    fi
  done

  {
    echo "finished_at=$(date -Is)"
    echo "converted=$done skipped=$skipped failed=$failed"
  } | tee -a "$log_file"
}

activate_ffmpeg
convert_tree "${ROOT}/generations" "generations"
convert_tree "${ROOT}/generations2" "generations2"
