#!/usr/bin/env python3
"""Overlay the resampled 240x432 AllTracker dense tracks onto each scene's clip.

Reads ``data_wan_1k/alltracker_dense_tracks/<scene>_240x432.npz``::

    flow        (T, 240, 432, 2)  float16  pixel displacement from query frame
    visibs      (T, 240, 432)     bool     visibility mask
    confs       (T, 240, 432)     float16  vis confidence
    image_size  (2,)              int32    [240, 432]
    query_frame ()                int32

and the matching RGB video ``data_wan_1k/videos/<scene>.mp4`` (1280x720) — NOT
``clips/`` (that's the DrRobot render). The tracks were extracted from the real
DROID camera RGB, so the overlay must go on the RGB. For every
stride-spaced query pixel it splats a colored dot at the tracked location with
a fading motion trail. Dot color is stable per-track (HSV of query xy), so
motion patterns are easy to follow. Writes ``<out_dir>/<scene>.mp4``.

Usage (from repo root):

  conda run --no-capture-output -n dr python scripts/viz_dense_tracks_240x432.py \
      --scenes scene_00001 scene_00004 scene_00100

  # first N scenes that have a _240x432 npz:
  conda run --no-capture-output -n dr python scripts/viz_dense_tracks_240x432.py --max_scenes 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DENSE_DIR = REPO / "data_wan_1k" / "alltracker_dense_tracks"
RGB_DIR = REPO / "data_wan_1k" / "videos"
OUT_DIR = REPO / "data_wan_1k" / "dense_tracks_viz"
SUFFIX = "_240x432"


def _read_frames(mp4: Path):
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {mp4}")
    return frames, float(fps), w, h


def _open_writer(out_path: Path, fps: float):
    import imageio.v2 as imageio
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(out_path), format="ffmpeg", fps=fps, codec="libx264",
        quality=8, pixelformat="yuv420p", macro_block_size=None,
        ffmpeg_params=["-movflags", "+faststart"],
    )


def viz_scene(scene: str, npz_path: Path, rgb_path: Path, out_path: Path,
              stride: int, trail_decay: float, bkg_opacity: float) -> None:
    frames, fps, vidW, vidH = _read_frames(rgb_path)

    z = np.load(npz_path)
    flow = z["flow"]                       # (T, H, W, 2) float16
    visibs = z["visibs"]                   # (T, H, W)    bool
    atH, atW = (int(x) for x in z["image_size"])

    sx, sy = vidW / float(atW), vidH / float(atH)
    T = min(len(frames), flow.shape[0])
    if len(frames) != flow.shape[0]:
        print(f"  [warn] {scene}: {len(frames)} clip frames vs {flow.shape[0]} "
              f"track frames -> using {T}")
    frames, flow, visibs = frames[:T], flow[:T], visibs[:T]

    s = max(int(stride), 1)
    ys = np.arange(0, atH, s, dtype=np.int32)
    xs = np.arange(0, atW, s, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy_flat, xx_flat = yy.reshape(-1), xx.reshape(-1)
    N = yy_flat.size
    qx_vid = xx_flat.astype(np.float32) * sx
    qy_vid = yy_flat.astype(np.float32) * sy

    u = xx_flat.astype(np.float32) / max(atW - 1, 1)
    v = yy_flat.astype(np.float32) / max(atH - 1, 1)
    hsv = np.stack([
        (u * 179.0).astype(np.uint8),
        np.full(N, 255, dtype=np.uint8),
        (v * 200.0 + 55.0).astype(np.uint8),
    ], axis=-1)[None, :, :]
    colors_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0].astype(np.float32)

    trail_rgb = np.zeros((vidH, vidW, 3), dtype=np.float32)
    trail_a = np.zeros((vidH, vidW), dtype=np.float32)

    writer = _open_writer(out_path, fps)
    try:
        for t in range(T):
            f_xy = flow[t, yy_flat, xx_flat, :].astype(np.float32)
            v_at = visibs[t, yy_flat, xx_flat]

            cx = (qx_vid + f_xy[:, 0] * sx + 0.5).astype(np.int32)
            cy = (qy_vid + f_xy[:, 1] * sy + 0.5).astype(np.int32)
            valid = v_at & (cx >= 0) & (cx < vidW) & (cy >= 0) & (cy < vidH)

            trail_rgb *= trail_decay
            trail_a *= trail_decay
            trail_rgb[cy[valid], cx[valid]] = colors_bgr[valid]
            trail_a[cy[valid], cx[valid]] = 1.0

            bg = frames[t].astype(np.float32) * bkg_opacity
            alpha3 = trail_a[..., None]
            out = np.clip(bg * (1.0 - alpha3) + trail_rgb * alpha3, 0, 255).astype(np.uint8)
            writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    print(f"  [save] {out_path}  ({T} frames @ {fps:.1f} fps, {vidW}x{vidH}, "
          f"N={N} pts/frame, stride={s})")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dense_dir", type=Path, default=DENSE_DIR)
    p.add_argument("--rgb_dir", type=Path, default=RGB_DIR,
                   help="Dir of RGB <scene>.mp4 (data_wan_1k/videos, NOT clips/).")
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    p.add_argument("--scenes", nargs="*", default=None,
                   help="Explicit scene ids (e.g. scene_00001). Default: scan dense_dir.")
    p.add_argument("--max_scenes", type=int, default=4,
                   help="When --scenes not given, process the first N (0 = all).")
    p.add_argument("--stride", type=int, default=2,
                   help="Pixel stride between drawn tracks at 240x432 res.")
    p.add_argument("--trail_decay", type=float, default=0.85)
    p.add_argument("--bkg_opacity", type=float, default=0.5)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing outputs (default: skip if present).")
    args = p.parse_args()

    if args.scenes:
        scenes = list(args.scenes)
    else:
        scenes = sorted(q.name[: -len(SUFFIX + ".npz")]
                        for q in args.dense_dir.glob(f"*{SUFFIX}.npz"))
        if args.max_scenes > 0:
            scenes = scenes[: args.max_scenes]
    if not scenes:
        raise SystemExit(f"no *{SUFFIX}.npz found under {args.dense_dir}")

    print(f"[plan] {len(scenes)} scene(s) -> {args.out_dir}  stride={args.stride}")
    ok = skip = fail = 0
    for i, scene in enumerate(scenes):
        npz = args.dense_dir / f"{scene}{SUFFIX}.npz"
        rgb = args.rgb_dir / f"{scene}.mp4"
        out = args.out_dir / f"{scene}.mp4"
        print(f"\n=== [{i + 1}/{len(scenes)}] {scene} ===")
        if not npz.is_file():
            print(f"  [skip] missing {npz}"); skip += 1; continue
        if not rgb.is_file():
            print(f"  [skip] missing {rgb}"); skip += 1; continue
        if out.is_file() and not args.force:
            print(f"  [skip] exists (use --force): {out}"); skip += 1; continue
        try:
            viz_scene(scene, npz, rgb, out, args.stride,
                      args.trail_decay, args.bkg_opacity)
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"  [FAIL] {scene}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n[done] ok={ok} skipped={skip} failed={fail} -> {args.out_dir}")


if __name__ == "__main__":
    main()
