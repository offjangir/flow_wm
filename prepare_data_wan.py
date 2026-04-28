#!/usr/bin/env python3
"""
Build ``data_wan/`` for render-conditioned + tracks-supervised world-model
training.

Per-scene data the world model consumes:

  - **Target** (what the DiT learns to denoise): real DROID camera RGB video
    (e.g. ext1 = serial ``20103212``). Symlinked into
    ``data_wan/videos/<scene>.mp4`` so all training paths are relative to
    ``data_wan/``.

  - **Conditioning** (per-frame action signal): DrRobot rendering of the
    robot executing the action. Already lives at
    ``data_wan/clips/<scene>.mp4`` (kept; no symlinking).

  - **Auxiliary supervision** (point tracks): AllTracker sparse trajectories
    extracted from the *real* DROID camera (NOT from the render). Already
    lives at ``data_wan/alltracker_tracks/<scene>.npz``.

This script:

  1. Symlinks each scene's real DROID MP4 into ``data_wan/videos/<scene>.mp4``.
  2. Verifies the corresponding ``clips/<scene>.mp4`` (DrRobot render) exists.
  3. Verifies the corresponding ``alltracker_tracks/<scene>.npz`` exists.
  4. Writes ``data_wan/metadata.csv`` with columns
     ``video, prompt, render, tracks`` (paths relative to ``data_wan/``).

Usage::

  # default: ext1 camera (= serial 20103212), all scenes
  python prepare_data_wan.py

  # different camera
  python prepare_data_wan.py --camera_key ext2

  # don't fail if tracks are missing for some scenes (still emit those rows
  # with an empty ``tracks`` column)
  python prepare_data_wan.py --allow_missing_tracks

  # only first 2 scenes, useful for smoke tests
  python prepare_data_wan.py --max_scenes 2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

SIDEGIG = Path(__file__).resolve().parent
DATA_WAN = SIDEGIG / "data_wan"
DROID_ROOT = SIDEGIG / "data" / "droid_10_demos"
RENDERS_DIR = DATA_WAN / "clips"            # DrRobot renderings (conditioning)
VIDEOS_DIR = DATA_WAN / "videos"            # symlinks to real DROID RGB
TRACKS_DIR = DATA_WAN / "alltracker_tracks" # AllTracker sparse tracks

CAMERA_KEYS = ("ext1", "ext2", "wrist")


# ---------------------------------------------------------------------------
# Scene + DROID metadata helpers
# ---------------------------------------------------------------------------

def get_scene_names() -> list[str]:
    return sorted(os.path.basename(d) for d in glob.glob(str(DROID_ROOT / "scene_*")))


def _load_droid_metadata(scene_name: str) -> dict | None:
    scene_dir = DROID_ROOT / scene_name
    jsons = list(scene_dir.glob("metadata_*.json"))
    if not jsons:
        return None
    with open(jsons[0], "r") as f:
        return json.load(f)


def _real_droid_mp4(scene_name: str, camera_key: str) -> Path | None:
    """Resolve real DROID camera serial -> MP4 path for one scene."""
    meta = _load_droid_metadata(scene_name)
    if meta is None:
        return None
    field = f"{camera_key}_cam_serial"
    serial = meta.get(field)
    if serial is None:
        return None
    mp4 = DROID_ROOT / scene_name / "recordings" / "MP4" / f"{serial}.mp4"
    return mp4 if mp4.exists() else None


def _prompt_for_scene(scene_name: str) -> str:
    """Caption from DROID ``current_task`` (lower-cased, prefixed)."""
    meta = _load_droid_metadata(scene_name)
    if meta and "current_task" in meta:
        raw = meta["current_task"]
        first_line = raw.strip().split("\n")[0].strip()
        if first_line.lower().startswith("do any task"):
            lines = [l.strip().lstrip("* ")
                     for l in raw.strip().split("\n") if l.strip().startswith("*")]
            first_line = lines[0] if lines else "object manipulation"
        return f"robot arm {first_line.lower()}"
    return "robot arm manipulation in a lab"


# ---------------------------------------------------------------------------
# Filesystem ops
# ---------------------------------------------------------------------------

def _relink(src: Path, dest: Path) -> None:
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dest)


def stage_symlink_videos(scene_names: list[str], camera_key: str) -> dict[str, Path]:
    """Symlink real DROID MP4s into ``data_wan/videos/<scene>.mp4``."""
    out: dict[str, Path] = {}
    for scene in scene_names:
        src = _real_droid_mp4(scene, camera_key)
        if src is None:
            print(f"  [SKIP video] {scene}: no DROID {camera_key} MP4")
            continue
        dest = VIDEOS_DIR / f"{scene}.mp4"
        _relink(src, dest)
        out[scene] = dest
        print(f"  [video ] {scene} -> {src.relative_to(SIDEGIG)}")
    return out


def collect_renders(scene_names: list[str]) -> dict[str, Path]:
    """DrRobot renderings already live at ``data_wan/clips/<scene>.mp4``."""
    out: dict[str, Path] = {}
    for scene in scene_names:
        p = RENDERS_DIR / f"{scene}.mp4"
        if p.exists():
            out[scene] = p
        else:
            print(f"  [SKIP render] {scene}: no render at {p.relative_to(SIDEGIG)}")
    return out


def collect_tracks(scene_names: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for scene in scene_names:
        p = TRACKS_DIR / f"{scene}.npz"
        if p.exists():
            out[scene] = p
        else:
            print(f"  [SKIP tracks] {scene}: no tracks at {p.relative_to(SIDEGIG)}")
    return out


# ---------------------------------------------------------------------------
# metadata.csv
# ---------------------------------------------------------------------------

def write_metadata(
    scene_names: list[str],
    videos: dict[str, Path],
    renders: dict[str, Path],
    tracks: dict[str, Path],
    allow_missing_tracks: bool,
) -> Path:
    csv_path = DATA_WAN / "metadata.csv"
    rows: list[dict[str, str]] = []
    skipped: list[str] = []
    for scene in scene_names:
        if scene not in videos or scene not in renders:
            skipped.append(scene)
            continue
        if scene not in tracks and not allow_missing_tracks:
            skipped.append(scene)
            continue
        rows.append({
            "video": os.path.relpath(videos[scene], DATA_WAN),
            "prompt": _prompt_for_scene(scene),
            "render": os.path.relpath(renders[scene], DATA_WAN),
            "tracks": (os.path.relpath(tracks[scene], DATA_WAN)
                       if scene in tracks else ""),
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt", "render", "tracks"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[metadata] wrote {len(rows)} rows -> {csv_path}")
    if skipped:
        print(f"[metadata] skipped {len(skipped)} scene(s) for missing assets: "
              + ", ".join(skipped[:5])
              + (" ..." if len(skipped) > 5 else ""))
    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--camera_key", type=str, default="ext1", choices=CAMERA_KEYS,
        help="DROID camera to use as the training target video.",
    )
    parser.add_argument(
        "--allow_missing_tracks", action="store_true",
        help="Emit a row even if alltracker_tracks/<scene>.npz is missing "
             "(empty `tracks` column). Default: skip such scenes.",
    )
    parser.add_argument(
        "--max_scenes", type=int, default=-1,
        help="Limit number of scenes processed (-1 = all).",
    )
    args = parser.parse_args()

    DATA_WAN.mkdir(parents=True, exist_ok=True)

    scenes = get_scene_names()
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    print(f"Processing {len(scenes)} scene(s)\n")

    print(f"=== Stage 1: symlink DROID {args.camera_key} cameras -> "
          f"data_wan/videos/ ===")
    videos = stage_symlink_videos(scenes, args.camera_key)

    print(f"\n=== Stage 2: collect DrRobot renders from data_wan/clips/ ===")
    if not RENDERS_DIR.is_dir():
        print(f"[ERROR] DrRobot renders dir not found: {RENDERS_DIR}")
        sys.exit(1)
    renders = collect_renders(scenes)

    print(f"\n=== Stage 3: collect AllTracker tracks from "
          f"data_wan/alltracker_tracks/ ===")
    if not TRACKS_DIR.is_dir():
        print(f"[ERROR] tracks dir not found: {TRACKS_DIR}\n"
              "Run: python scripts/extract_alltracker.py")
        sys.exit(1)
    tracks = collect_tracks(scenes)

    print(f"\n=== Stage 4: write metadata.csv ===")
    write_metadata(scenes, videos, renders, tracks, args.allow_missing_tracks)

    print("\n=== Summary ===")
    print(f"  data_wan/videos/             : {len(videos)} target videos "
          f"(DROID {args.camera_key})")
    print(f"  data_wan/clips/              : {len(renders)} render conditioning "
          f"videos (DrRobot)")
    print(f"  data_wan/alltracker_tracks/  : {len(tracks)} track .npz files")
    print(f"  data_wan/metadata.csv        : ready for training")
    print(f"\nNext:\n  python -m world_model.wan_flow.train "
          f"--config configs/train_drrobot.json")


if __name__ == "__main__":
    main()
