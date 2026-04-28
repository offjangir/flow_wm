#!/usr/bin/env python3
"""
Overlay AllTracker point tracks onto each scene's real DROID camera MP4
(``data/droid_10_demos/<scene>/recordings/MP4/<camera>.mp4``) to visualize
how points move through time.

This script reads PRE-SAVED tracks (does NOT re-run AllTracker). It supports
two saved formats and prefers the dense one when present:

  1. DENSE per-pixel tracks (``data_wan/alltracker_dense_tracks/<scene>.npz``)::

         flow        (T, H, W, 2)  float16  pixel displacement from query frame
         visibs      (T, H, W)     bool     visibility mask
         confs       (T, H, W)     float16  raw vis-conf (pre-threshold)
         image_size  (2,)          int32    [H, W] AllTracker ran at

     → produced by ``scripts/extract_alltracker.py --save_dense_tracks``.
     This gives a track for EVERY pixel of the query frame; the viz draws a
     colored dot at the current position of every (stride-spaced) pixel,
     with a fading motion trail.

  2. SPARSE point tracks (``data_wan/alltracker_tracks/<scene>.npz``)::

         trajs       (T, N, 2)  float32  xy in AllTracker resolution
         visibs      (T, N)     bool
         queries_xy0 (N, 2)     float32  query-frame xy (used for stable colors)
         image_size  (2,)       int32

     → produced by the default ``scripts/extract_alltracker.py`` run
     (32×20 = 640 query points). Used as a fallback; viz draws polyline
     trails + dots per track.

Both modes write to ``data_wan/tracks_viz/<scene>.mp4``.

Track colors are stable per-track and derived from the query-frame ``(x, y)``
(an HSV gradient), so motion patterns are easy to follow visually.

Usage (from ``wm/`` root, in the ``dr`` conda env)::

  # all scenes — auto-prefers dense .npz when present:
  python scripts/viz_tracks.py

  # force the sparse fallback even if dense exists:
  python scripts/viz_tracks.py --no-prefer_dense

  # dense viz at stride 2 (default), bigger trail decay:
  python scripts/viz_tracks.py --dense_stride 2 --trail_decay 0.8

  # quick test (one scene):
  python scripts/viz_tracks.py --max_scenes 1 --force

  # different DROID camera serial:
  python scripts/viz_tracks.py --droid_camera 16787047

  # fall back to the resampled data_wan/clips/ mp4s (frame counts may differ):
  python scripts/viz_tracks.py --no-use_droid_root
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

SIDEGIG = Path(__file__).resolve().parent.parent
DATA_WAN = SIDEGIG / "data_wan"
CLIPS_DIR = DATA_WAN / "clips"
TRACKS_DIR = DATA_WAN / "alltracker_tracks"
DENSE_TRACKS_DIR = DATA_WAN / "alltracker_dense_tracks"
OUT_DIR = DATA_WAN / "tracks_viz"
DROID_ROOT = SIDEGIG / "data" / "droid_10_demos"
DEFAULT_DROID_CAMERA = "20103212"


def _resolve_source_mp4(
    scene: str,
    use_droid_root: bool,
    droid_root: Path,
    droid_camera: str,
    clips_dir: Path,
) -> Path | None:
    """Return the source MP4 the tracks were computed from for this scene."""
    if use_droid_root:
        p = droid_root / scene / "recordings" / "MP4" / f"{droid_camera}.mp4"
        return p if p.is_file() else None
    p = clips_dir / f"{scene}.mp4"
    return p if p.is_file() else None


def _read_video_frames(mp4_path: Path) -> tuple[List[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {mp4_path}")
    return frames, float(fps), w, h


def _track_colors_bgr(queries_xy0: np.ndarray, atH: int, atW: int) -> np.ndarray:
    """Stable per-track colors (N, 3) uint8 BGR derived from query-frame xy."""
    u = np.clip(queries_xy0[:, 0] / max(atW - 1, 1), 0.0, 1.0)
    v = np.clip(queries_xy0[:, 1] / max(atH - 1, 1), 0.0, 1.0)
    hsv = np.stack(
        [
            (u * 179.0).astype(np.uint8),
            np.full_like(u, 255, dtype=np.uint8),
            (v * 200.0 + 55.0).astype(np.uint8),
        ],
        axis=-1,
    )[None, :, :]
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
    return bgr


def _open_writer(out_path: Path, fps: float, w: int, h: int):
    """Open an H.264/yuv420p mp4 writer that plays in browsers/Cursor."""
    import imageio.v2 as imageio
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(out_path),
        format="ffmpeg",
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-movflags", "+faststart"],
    )


def _process_scene_dense(
    scene: str,
    clip_path: Path,
    dense_npz_path: Path,
    out_path: Path,
    stride: int,
    trail_decay: float,
    bkg_opacity: float,
) -> None:
    """Per-pixel track viz from a dense ``flow (T,H,W,2)`` + ``visibs (T,H,W)``
    npz. Splats every (stride-spaced) query-frame pixel onto each frame at its
    tracked location, with a fading motion trail accumulated on a canvas.
    """
    frames, fps, vidW, vidH = _read_video_frames(clip_path)

    z = np.load(dense_npz_path)
    flow = z["flow"]                  # (T, H, W, 2) float16
    visibs = z["visibs"]              # (T, H, W)    bool
    atH, atW = (int(x) for x in z["image_size"])

    sx, sy = vidW / float(atW), vidH / float(atH)

    T = min(len(frames), flow.shape[0])
    frames = frames[:T]
    flow = flow[:T]
    visibs = visibs[:T]

    s = max(int(stride), 1)
    ys = np.arange(0, atH, s, dtype=np.int32)
    xs = np.arange(0, atW, s, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy_flat = yy.reshape(-1)          # (N,)
    xx_flat = xx.reshape(-1)          # (N,)
    N = yy_flat.size
    qx_vid = xx_flat.astype(np.float32) * sx
    qy_vid = yy_flat.astype(np.float32) * sy

    u = xx_flat.astype(np.float32) / max(atW - 1, 1)
    v = yy_flat.astype(np.float32) / max(atH - 1, 1)
    hsv = np.stack(
        [
            (u * 179.0).astype(np.uint8),
            np.full(N, 255, dtype=np.uint8),
            (v * 200.0 + 55.0).astype(np.uint8),
        ],
        axis=-1,
    )[None, :, :]
    colors_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0].astype(np.float32)  # (N, 3)

    trail_rgb = np.zeros((vidH, vidW, 3), dtype=np.float32)
    trail_a = np.zeros((vidH, vidW), dtype=np.float32)

    writer = _open_writer(out_path, fps, vidW, vidH)
    try:
        for t in range(T):
            f_xy = flow[t, yy_flat, xx_flat, :].astype(np.float32)   # (N, 2)
            v_at = visibs[t, yy_flat, xx_flat]                       # (N,) bool

            cx = (qx_vid + f_xy[:, 0] * sx + 0.5).astype(np.int32)
            cy = (qy_vid + f_xy[:, 1] * sy + 0.5).astype(np.int32)

            valid = v_at & (cx >= 0) & (cx < vidW) & (cy >= 0) & (cy < vidH)
            cx_v = cx[valid]
            cy_v = cy[valid]
            col_v = colors_bgr[valid]

            trail_rgb *= trail_decay
            trail_a *= trail_decay
            trail_rgb[cy_v, cx_v] = col_v
            trail_a[cy_v, cx_v] = 1.0

            bg = frames[t].astype(np.float32) * bkg_opacity
            alpha3 = trail_a[..., None]
            frame_out = bg * (1.0 - alpha3) + trail_rgb * alpha3
            frame_out = np.clip(frame_out, 0, 255).astype(np.uint8)
            writer.append_data(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    print(
        f"  [save dense] {out_path}  ({T} frames @ {fps:.1f} fps, {vidW}x{vidH}, "
        f"N={N} pts/frame, stride={s})"
    )


def _process_scene(
    scene: str,
    clip_path: Path,
    npz_path: Path,
    out_path: Path,
    trail: int,
    radius: int,
    line_thickness: int,
) -> None:
    frames, fps, vidW, vidH = _read_video_frames(clip_path)

    z = np.load(npz_path)
    trajs = z["trajs"].astype(np.float32)            # (T, N, 2)
    visibs = z["visibs"].astype(bool)                # (T, N)
    queries_xy0 = z["queries_xy0"].astype(np.float32)  # (N, 2)
    atH, atW = (int(x) for x in z["image_size"])

    sx, sy = vidW / float(atW), vidH / float(atH)
    trajs[..., 0] *= sx
    trajs[..., 1] *= sy
    queries_xy0[:, 0] *= sx
    queries_xy0[:, 1] *= sy

    T = min(len(frames), trajs.shape[0])
    frames = frames[:T]
    trajs = trajs[:T]
    visibs = visibs[:T]

    colors_bgr = _track_colors_bgr(queries_xy0, vidH, vidW)
    colors_list = [tuple(int(c) for c in colors_bgr[n]) for n in range(colors_bgr.shape[0])]

    writer = _open_writer(out_path, fps, vidW, vidH)

    try:
        N = trajs.shape[1]
        for t in range(T):
            frame = frames[t].copy()
            t0 = max(0, t - trail)
            for n in range(N):
                if not visibs[t, n]:
                    continue
                vis_window = visibs[t0 : t + 1, n]
                pts_window = trajs[t0 : t + 1, n][vis_window]
                if pts_window.shape[0] >= 2:
                    pts = pts_window.astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        frame,
                        [pts],
                        isClosed=False,
                        color=colors_list[n],
                        thickness=line_thickness,
                        lineType=cv2.LINE_AA,
                    )
                cx, cy = trajs[t, n]
                cv2.circle(
                    frame,
                    (int(cx), int(cy)),
                    radius=radius,
                    color=colors_list[n],
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    print(f"  [save] {out_path}  ({T} frames @ {fps:.1f} fps, {vidW}x{vidH})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--droid_root", type=Path, default=DROID_ROOT,
                   help="DROID dataset root: <root>/<scene>/recordings/MP4/<cam>.mp4.")
    p.add_argument("--droid_camera", type=str, default=DEFAULT_DROID_CAMERA,
                   help="DROID camera serial that the tracks were extracted from.")
    p.add_argument("--use_droid_root", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Pull source mp4s from --droid_root (default). Pass "
                        "--no-use_droid_root to fall back to --clips_dir.")
    p.add_argument("--clips_dir", type=Path, default=CLIPS_DIR,
                   help="Fallback flat directory of <scene>.mp4 (only used when "
                        "--no-use_droid_root is passed).")
    p.add_argument("--tracks_dir", type=Path, default=TRACKS_DIR,
                   help="Directory of sparse-track .npz files.")
    p.add_argument("--dense_tracks_dir", type=Path, default=DENSE_TRACKS_DIR,
                   help="Directory of dense per-pixel-track .npz files (preferred "
                        "when present).")
    p.add_argument("--prefer_dense", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="If a dense .npz exists for a scene, use it (default). "
                        "Pass --no-prefer_dense to force the sparse fallback.")
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    p.add_argument("--dense_stride", type=int, default=2,
                   help="Pixel stride between drawn tracks in dense mode "
                        "(at AllTracker resolution). 1 = every pixel.")
    p.add_argument("--trail_decay", type=float, default=0.85,
                   help="Per-frame multiplier on the trail canvas in dense mode "
                        "(closer to 1 = longer trails).")
    p.add_argument("--bkg_opacity", type=float, default=0.5,
                   help="Background-frame opacity behind the dense overlay.")
    p.add_argument("--trail", type=int, default=24,
                   help="[sparse mode] Past frames to draw as a motion trail.")
    p.add_argument("--radius", type=int, default=3,
                   help="[sparse mode] Filled-circle radius for the current "
                        "track position.")
    p.add_argument("--line_thickness", type=int, default=1,
                   help="[sparse mode] Trail polyline thickness in px.")
    p.add_argument("--max_scenes", type=int, default=0,
                   help="Process only the first N scenes (0 = all).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing outputs (default: skip if present).")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sparse_npzs = sorted(args.tracks_dir.glob("*.npz"))
    dense_npzs = (
        sorted(args.dense_tracks_dir.glob("*.npz"))
        if args.dense_tracks_dir.is_dir() else []
    )
    sparse_by_scene = {p.stem: p for p in sparse_npzs}
    dense_by_scene = {p.stem: p for p in dense_npzs}

    scenes = sorted(set(sparse_by_scene) | set(dense_by_scene))
    if not scenes:
        raise SystemExit(
            f"no .npz tracks found under {args.tracks_dir} "
            f"or {args.dense_tracks_dir}"
        )
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    src_label = (
        f"DROID {args.droid_root} / *cam={args.droid_camera}.mp4"
        if args.use_droid_root else f"clips {args.clips_dir}"
    )
    print(
        f"[plan] {len(scenes)} scene(s) | source={src_label} | "
        f"prefer_dense={args.prefer_dense} dense_stride={args.dense_stride} "
        f"-> {args.out_dir}"
    )

    ok = skipped = failed = 0
    for i, scene in enumerate(scenes):
        src_mp4 = _resolve_source_mp4(
            scene, args.use_droid_root, args.droid_root,
            args.droid_camera, args.clips_dir,
        )
        out = args.out_dir / f"{scene}.mp4"
        print(f"\n=== [{i + 1}/{len(scenes)}] {scene} ===")
        if src_mp4 is None:
            print(f"  [skip] no source mp4 found for {scene}")
            skipped += 1
            continue
        if out.is_file() and not args.force:
            print(f"  [skip] exists (use --force): {out}")
            skipped += 1
            continue
        dense_npz = dense_by_scene.get(scene)
        sparse_npz = sparse_by_scene.get(scene)
        use_dense = args.prefer_dense and dense_npz is not None
        try:
            if use_dense:
                _process_scene_dense(
                    scene, src_mp4, dense_npz, out,
                    stride=args.dense_stride,
                    trail_decay=args.trail_decay,
                    bkg_opacity=args.bkg_opacity,
                )
            elif sparse_npz is not None:
                _process_scene(
                    scene, src_mp4, sparse_npz, out,
                    trail=args.trail, radius=args.radius,
                    line_thickness=args.line_thickness,
                )
            else:
                print(f"  [skip] no sparse .npz for {scene}")
                skipped += 1
                continue
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {scene}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n[done] ok={ok} skipped={skipped} failed={failed} -> {args.out_dir}")


if __name__ == "__main__":
    main()
