"""Rebuild data_wan/clips/<scene>.mp4 from data_wan/frames/<scene>/frame_*.jpg.

Use when clips/ contains broken symlinks but frames/ has the source JPEGs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def rebuild_one(scene_dir: Path, out_path: Path, fps: int, overwrite: bool) -> None:
    frames = sorted(p for p in scene_dir.iterdir() if p.suffix.lower() == ".jpg")
    if not frames:
        print(f"skip (no jpgs): {scene_dir}")
        return
    if out_path.exists() or out_path.is_symlink():
        if not overwrite:
            print(f"skip (exists): {out_path}")
            return
        out_path.unlink()

    writer = imageio.get_writer(
        str(out_path), format="ffmpeg", fps=fps,
        codec="libx264", quality=8, macro_block_size=None,
    )
    try:
        for f in frames:
            arr = np.asarray(Image.open(f).convert("RGB"))
            writer.append_data(arr)
    finally:
        writer.close()
    print(f"wrote {out_path} ({len(frames)} frames, {fps} fps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_wan", help="data_wan directory (relative or absolute)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--no-overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    frames_root = root / "frames"
    clips_root = root / "clips"
    if not frames_root.is_dir():
        print(f"error: {frames_root} is not a directory", file=sys.stderr)
        sys.exit(1)
    clips_root.mkdir(exist_ok=True)

    scene_dirs = sorted(p for p in frames_root.iterdir() if p.is_dir())
    if not scene_dirs:
        print(f"error: no scene dirs under {frames_root}", file=sys.stderr)
        sys.exit(1)

    for sd in scene_dirs:
        out = clips_root / f"{sd.name}.mp4"
        if out.is_symlink() and not out.exists():
            out.unlink()
        rebuild_one(sd, out, fps=args.fps, overwrite=not args.no_overwrite)


if __name__ == "__main__":
    main()
