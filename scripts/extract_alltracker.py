#!/usr/bin/env python3
"""
Precompute AllTracker (Harley et al., ICCV 2025 — https://github.com/aharley/alltracker)
**sparse** point tracks on the **real DROID camera** videos as auxiliary
supervision for the world model.

Source video selection (default = real DROID camera, NOT DrRobot renders)::

  --droid_root <path>     scene root containing <scene>/recordings/MP4/<serial>.mp4
  --droid_camera <serial> camera serial (filename stem). Defaults to 20103212
                          (= DROID ext1 cam in this repo).

For each scene we read its real-camera MP4, run AllTracker's ``forward_sliding``
from ``--query_frame``, sample a uniform grid of query points at the query
frame, and save *sparse* per-point trajectories.

Outputs (per scene)::

  data_wan/alltracker_tracks/<scene>.npz
      trajs        (T, N, 2) float32   pixel xy at AllTracker working res
      visibs       (T, N)    bool      per-frame visibility (conf > conf_thr)
      confs        (T, N)    float32   raw visibility-confidence per frame
      queries_xy0  (N, 2)    float32   xy at the query frame
      image_size   (2,)      int32     [H, W] AllTracker ran at
      query_frame              int32   reference frame index
      conf_thr                 float32 threshold used to binarize `visibs`
      fps                      int32   source FPS

  data_wan/alltracker_viz/<scene>.mp4   (optional, --no_viz to skip)
      hstack: input | flow-color | sparse-track overlay (drawn at --viz_stride)

This sparse format (~1 MB / scene) is what the world-model auxiliary
track-prediction head consumes. Dense per-pixel flow output has been removed
from defaults; pass --save_dense_flow to additionally write a (T,H,W,2) .npy
under data_wan/alltracker_flow/ (large: ~500 MB / scene).

Usage (from ``wm/`` root, conda env ``dr``)::

  # full run, real DROID ext1 camera (20103212), default image_size 1024:
  python scripts/extract_alltracker.py

  # ext2 instead of ext1:
  python scripts/extract_alltracker.py --droid_camera 20655732

  # denser query grid (default 32x20 = 640 points to match prior outputs):
  python scripts/extract_alltracker.py --track_grid_x 48 --track_grid_y 30

  # additionally save the dense (T,H,W,2) flow npy:
  python scripts/extract_alltracker.py --save_dense_flow

  # also augment data_wan/metadata.csv with the `tracks` column:
  python scripts/extract_alltracker.py --update_metadata

  # legacy flat-clips-dir mode (e.g. tracks-of-renders ablation):
  python scripts/extract_alltracker.py --no-use_droid_root \\
      --clips_dir data_wan/clips

Requires a working AllTracker checkout at ``--alltracker_root`` that contains
``nets/alltracker.py`` and ``utils/{basic,improc,saveload}.py``. If the local
``/data/yjangir/sidegig/alltracker`` is empty, clone it first::

    git clone https://github.com/aharley/alltracker.git /data/yjangir/sidegig/alltracker
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SIDEGIG = Path(__file__).resolve().parent.parent
DATA_WAN = SIDEGIG / "data_wan"
CLIPS_DIR = DATA_WAN / "clips"
DROID_ROOT = SIDEGIG / "data" / "droid_10_demos"
OUT_FLOW_DIR = DATA_WAN / "alltracker_flow"
OUT_TRACKS_DIR = DATA_WAN / "alltracker_tracks"
OUT_DENSE_TRACKS_DIR = DATA_WAN / "alltracker_dense_tracks"
OUT_VIZ_DIR = DATA_WAN / "alltracker_viz"
METADATA_CSV = DATA_WAN / "metadata.csv"


def _enumerate_droid_clips(droid_root: Path, camera: str) -> List[Tuple[str, Path]]:
    """Walk ``<droid_root>/scene_*/recordings/MP4/<camera>.mp4`` and return a
    list of ``(scene_name, mp4_path)`` tuples. ``scene_name`` is the scene-dir
    basename (e.g. ``scene_Tue_May__9_01:17:11_2023``), which is how the
    outputs are named in ``data_wan/alltracker_tracks/`` and
    ``data_wan/alltracker_viz/``.
    """
    if not droid_root.is_dir():
        raise SystemExit(f"--droid_root {droid_root} is not a directory")
    out: List[Tuple[str, Path]] = []
    for scene_dir in sorted(droid_root.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue
        mp4 = scene_dir / "recordings" / "MP4" / f"{camera}.mp4"
        if not mp4.is_file():
            print(f"  [skip] {scene_dir.name}: missing {mp4}")
            continue
        out.append((scene_dir.name, mp4))
    if not out:
        raise SystemExit(
            f"no DROID clips matched {droid_root}/<scene>/recordings/MP4/{camera}.mp4"
        )
    return out


# ---------------------------------------------------------------------------
# AllTracker import bootstrap
# ---------------------------------------------------------------------------

def _import_alltracker(alltracker_root: Path):
    """Insert ``alltracker_root`` onto ``sys.path`` and import ``Net`` + utils.

    Returns ``(Net, utils_improc, utils_basic, utils_saveload)``. The ``utils_*``
    modules may be ``None`` if AllTracker's ``utils/`` is missing (we fall back
    to local flow-visualization + manual checkpoint loading).
    """
    root = alltracker_root.resolve()
    if not (root / "nets" / "alltracker.py").is_file():
        raise SystemExit(
            f"AllTracker repo not found at {root}\n"
            f"  missing {root / 'nets' / 'alltracker.py'}\n"
            "Clone it first:\n"
            f"  git clone https://github.com/aharley/alltracker.git {root}\n"
            "or pass --alltracker_root /path/to/alltracker."
        )
    sys.path.insert(0, str(root))
    from nets.alltracker import Net  # type: ignore

    def _try_import(name):
        try:
            return __import__(name, fromlist=["*"])
        except Exception as exc:
            print(f"[warn] could not import alltracker {name}: {exc}")
            return None

    return (
        Net,
        _try_import("utils.improc"),
        _try_import("utils.basic"),
        _try_import("utils.saveload"),
    )


# ---------------------------------------------------------------------------
# Video IO / resizing
# ---------------------------------------------------------------------------

def _read_mp4_frames(mp4_path: Path, max_frames: Optional[int]) -> Tuple[List[np.ndarray], int]:
    import cv2
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or 15
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {mp4_path}")
    return frames, fps


def _resize_keep_ar(frames: List[np.ndarray], image_size: int) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    import cv2
    h, w = frames[0].shape[:2]
    scale = min(image_size / h, image_size / w)
    new_h, new_w = (int(h * scale) // 8) * 8, (int(w * scale) // 8) * 8
    if (new_h, new_w) == (h, w):
        return frames, (h, w)
    out = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for f in frames]
    return out, (new_h, new_w)


def _resize_flow(flow_thw2: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize ``(T, H, W, 2)`` flow and rescale magnitudes."""
    import torch
    import torch.nn.functional as F
    _, h, w, _ = flow_thw2.shape
    if (h, w) == (out_h, out_w):
        return flow_thw2
    sx, sy = out_w / w, out_h / h
    x = torch.from_numpy(flow_thw2).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    out = x.permute(0, 2, 3, 1).numpy().astype(np.float32)
    out[..., 0] *= sx
    out[..., 1] *= sy
    return out


# ---------------------------------------------------------------------------
# Flow visualization (fallback if utils.improc.flow2color is unavailable)
# ---------------------------------------------------------------------------

def _flow_to_color_fallback(flow_hw2: np.ndarray) -> np.ndarray:
    import cv2
    mag, ang = cv2.cartToPolar(flow_hw2[..., 0], flow_hw2[..., 1])
    hsv = np.zeros((*flow_hw2.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * (180.0 / np.pi / 2.0)).astype(np.uint8)
    hsv[..., 1] = 255
    denom = max(np.percentile(mag, 99.0), 1e-3)
    hsv[..., 2] = np.clip(mag * (255.0 / denom), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _flow2color_batch(flows_e: "torch.Tensor", utils_improc) -> np.ndarray:
    _, T, _, H, W = flows_e.shape
    out = np.zeros((T, H, W, 3), dtype=np.uint8)
    if utils_improc is not None and hasattr(utils_improc, "flow2color"):
        for ti in range(T):
            viz = utils_improc.flow2color(flows_e[0:1, ti])
            out[ti] = viz[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        return out
    flows_np = flows_e[0].permute(0, 2, 3, 1).detach().cpu().numpy()
    for ti in range(T):
        out[ti] = _flow_to_color_fallback(flows_np[ti])
    return out


# ---------------------------------------------------------------------------
# Point-track overlay (adapted from AllTracker's demo.py)
# ---------------------------------------------------------------------------

def _draw_pts_gpu(
    rgbs: "torch.Tensor",   # (T, 3, H, W) float 0-255 on cuda
    trajs: "torch.Tensor",  # (T, N, 2)
    visibs: "torch.Tensor", # (T, N) bool
    colormap: np.ndarray,   # (N, 3) float
    radius: int = 2,
    bkg_opacity: float = 0.5,
    opacity: float = 0.9,
) -> np.ndarray:
    import torch
    device = rgbs.device
    T, C, H, W = rgbs.shape
    trajs = trajs.permute(1, 0, 2)  # (N,T,2)
    visibs = visibs.permute(1, 0)   # (N,T)
    colors = torch.as_tensor(colormap, dtype=torch.float32, device=device)

    rgbs = rgbs * bkg_opacity
    sharpness = 0.15
    D = radius * 2 + 1
    y = torch.arange(D, device=device).float()[:, None] - radius
    x = torch.arange(D, device=device).float()[None, :] - radius
    dist2 = x * x + y * y
    icon = torch.clamp(1 - (dist2 - (radius ** 2) / 2.0) / (radius * 2 * sharpness), 0, 1)
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")

    for t in range(T):
        mask = visibs[:, t]
        if mask.sum() == 0:
            continue
        xy = trajs[mask, t] + 0.5
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)
        colors_now = colors[mask]
        N = xy.shape[0]
        cx = xy[:, 0].long()
        cy = xy[:, 1].long()
        x_grid = cx[:, None, None] + disp_x
        y_grid = cy[:, None, None] + disp_y
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid = x_grid[valid]
        y_valid = y_grid[valid]
        icon_weights = icon.expand(N, D, D)[valid]
        colors_valid = (
            colors_now[:, :, None, None].expand(N, 3, D, D).permute(1, 0, 2, 3)[:, valid]
        )
        idx_flat = (y_valid * W + x_valid).long()

        accum = torch.zeros_like(rgbs[t])
        weight = torch.zeros(1, H * W, device=device)
        img_flat = accum.view(C, -1)
        weighted_colors = colors_valid * icon_weights
        img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors)
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W)
        alpha = weight.clamp(0, 1) * opacity
        accum = accum / (weight + 1e-6)
        rgbs[t] = rgbs[t] * (1 - alpha) + accum * alpha
    return rgbs.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def _get_2d_colors(xy0: np.ndarray, H: int, W: int, utils_improc) -> np.ndarray:
    # NOTE: we deliberately don't use ``utils_improc.get_2d_colors`` — it loads
    # ``./utils/bremm.png`` with a cwd-relative path that only resolves when
    # cwd == alltracker/. The HSV fallback below is colormap-agnostic.
    import cv2
    u = np.clip(xy0[:, 0] / max(W - 1, 1), 0, 1)
    v = np.clip(xy0[:, 1] / max(H - 1, 1), 0, 1)
    hsv = np.stack([
        (u * 179).astype(np.uint8),
        np.full_like(u, 255, dtype=np.uint8),
        (v * 200 + 55).astype(np.uint8),
    ], axis=-1)
    return cv2.cvtColor(hsv[None, :, :], cv2.COLOR_HSV2RGB)[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Dense per-pixel tracks from AllTracker flow + visconf
# ---------------------------------------------------------------------------

def _dense_tracks_from_flow(
    flows_e: "torch.Tensor",    # (1, T, 2, H, W)  pixel displacement from query_frame
    visconf_e: "torch.Tensor",  # (1, T, 2, H, W)  channel 1 = visibility conf
    conf_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Package the full dense AllTracker output as per-pixel tracks.

    Each pixel ``(x, y)`` of the query frame defines a track; its position at
    frame ``t`` is ``(x, y) + flow[t, :, y, x]``. To keep disk + memory small
    we store only the **offsets** (flow, in float16) plus a bool visibility
    mask; the dataloader reconstructs absolute positions as needed.

    Returns ``(flow_thw2 float16, visibs_thw bool, confs_thw float16)``.
    """
    # (1, T, 2, H, W) -> (T, H, W, 2)
    flow_thw2 = flows_e[0].permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float16)
    vis_conf = visconf_e[0, :, 1].detach()  # (T, H, W)
    visibs_thw = (vis_conf > conf_thr).cpu().numpy()
    confs_thw = vis_conf.cpu().numpy().astype(np.float16)
    return flow_thw2, visibs_thw, confs_thw


def _sparse_query_grid(H: int, W: int, grid_x: int, grid_y: int) -> np.ndarray:
    """Uniform (grid_y, grid_x) lattice of query xy in [0, W-1] x [0, H-1].

    Returned as ``(N, 2)`` float32 with N = grid_x * grid_y.
    """
    xs = np.linspace(0, W - 1, grid_x, dtype=np.float32)
    ys = np.linspace(0, H - 1, grid_y, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx, yy], axis=-1).reshape(-1, 2)


def _sparse_tracks_from_flow(
    flows_e: "torch.Tensor",    # (1, T, 2, H, W)  pixel displacement from query_frame
    visconf_e: "torch.Tensor",  # (1, T, 2, H, W)  channel 1 = visibility conf
    queries_xy0: np.ndarray,    # (N, 2) float, xy at query_frame in (W, H) coords
    conf_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample dense AllTracker outputs at the query points to get sparse tracks.

    Returns ``(trajs (T,N,2) f32, visibs (T,N) bool, confs (T,N) f32)``.
    Bilinear interpolation in xy (sub-pixel queries are supported).
    """
    import torch
    import torch.nn.functional as F

    _, T, _, H, W = flows_e.shape
    N = queries_xy0.shape[0]

    q = torch.from_numpy(queries_xy0).to(flows_e.device, dtype=flows_e.dtype)  # (N,2)
    # Normalize to grid_sample's [-1, 1].
    qx = (q[:, 0] / max(W - 1, 1)) * 2.0 - 1.0
    qy = (q[:, 1] / max(H - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([qx, qy], dim=-1).view(1, 1, N, 2).expand(T, 1, N, 2)

    flow_t = flows_e[0]                                  # (T, 2, H, W)
    vis_t = visconf_e[0, :, 1:2]                         # (T, 1, H, W)

    flow_at_q = F.grid_sample(flow_t, grid, mode="bilinear", align_corners=True)
    flow_at_q = flow_at_q.squeeze(2).permute(0, 2, 1).contiguous()   # (T, N, 2)
    conf_at_q = F.grid_sample(vis_t, grid, mode="bilinear", align_corners=True)
    conf_at_q = conf_at_q.squeeze(2).squeeze(1).contiguous()         # (T, N)

    trajs = (flow_at_q + q.view(1, N, 2)).detach().cpu().numpy().astype(np.float32)
    confs = conf_at_q.detach().cpu().numpy().astype(np.float32)
    visibs = confs > conf_thr
    return trajs, visibs, confs


def _viz_stride_grid(H: int, W: int, stride: int) -> np.ndarray:
    """``(N, 2)`` int (x, y) pixel coords on a stride-spaced grid (for viz only)."""
    xs = np.arange(0, W, max(stride, 1), dtype=np.int64)
    ys = np.arange(0, H, max(stride, 1), dtype=np.int64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx, yy], axis=-1).reshape(-1, 2)


# ---------------------------------------------------------------------------
# MP4 writer
# ---------------------------------------------------------------------------

def _write_mp4(path: Path, frames_rgb: np.ndarray, fps: int) -> None:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path), format="ffmpeg", fps=fps,
        codec="libx264", quality=8, macro_block_size=None,
    )
    try:
        for fr in frames_rgb:
            writer.append_data(fr)
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# AllTracker forward (sliding forward + optional backward for query_frame > 0)
# ---------------------------------------------------------------------------

def _forward_alltracker(
    model,
    rgbs: "torch.Tensor",  # (1, T, 3, H, W) float on cuda
    query_frame: int,
    inference_iters: int,
):
    import torch
    flows_fw, visconf_fw, _, _ = model.forward_sliding(
        rgbs[:, query_frame:], iters=inference_iters, sw=None, is_training=False,
    )
    if query_frame == 0:
        return flows_fw.cuda(), visconf_fw.cuda()
    flows_bw, visconf_bw, _, _ = model.forward_sliding(
        rgbs[:, : query_frame + 1].flip([1]),
        iters=inference_iters, sw=None, is_training=False,
    )
    flows_bw = flows_bw.cuda().flip([1])[:, :-1]
    visconf_bw = visconf_bw.cuda().flip([1])[:, :-1]
    return (
        torch.cat([flows_bw, flows_fw.cuda()], dim=1),
        torch.cat([visconf_bw, visconf_fw.cuda()], dim=1),
    )


# ---------------------------------------------------------------------------
# Per-clip worker
# ---------------------------------------------------------------------------

def process_clip(
    model,
    mp4_path: Path,
    out_flow_path: Path,
    out_tracks_path: Path,
    out_dense_tracks_path: Path,
    out_viz_path: Path,
    args,
    utils_improc,
    utils_basic,
) -> None:
    import torch

    frames, fps = _read_mp4_frames(mp4_path, args.max_frames)
    frames, (H, W) = _resize_keep_ar(frames, args.image_size)
    T = len(frames)
    q = max(0, min(args.query_frame, T - 1))

    rgbs = [torch.from_numpy(f).permute(2, 0, 1) for f in frames]
    rgbs = torch.stack(rgbs, dim=0).unsqueeze(0).float().cuda()  # (1, T, 3, H, W)

    t0 = time.time()
    flows_e, visconf_e = _forward_alltracker(model, rgbs, q, args.inference_iters)
    dt = time.time() - t0
    print(f"  [fwd] {T} frames @ {H}x{W} in {dt:.1f}s ({T / max(dt, 1e-6):.1f} fps)")

    # ---- sparse point tracks on a uniform query grid at the query frame ----
    queries_xy0 = _sparse_query_grid(H, W, args.track_grid_x, args.track_grid_y)
    trajs, visibs, confs = _sparse_tracks_from_flow(
        flows_e, visconf_e, queries_xy0, args.track_conf_thr,
    )  # (T, N, 2) f32 | (T, N) bool | (T, N) f32

    if args.save_tracks:
        out_tracks_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_tracks_path,
            trajs=trajs,                  # (T, N, 2) float32  pixel xy
            visibs=visibs,                # (T, N)    bool     vis (conf > thr)
            confs=confs,                  # (T, N)    float32  raw vis-conf
            queries_xy0=queries_xy0,      # (N, 2)    float32  xy at query_frame
            image_size=np.array([H, W], dtype=np.int32),
            query_frame=np.int32(q),
            conf_thr=np.float32(args.track_conf_thr),
            fps=np.int32(fps),
        )
        vis_pct = 100.0 * visibs.sum() / max(visibs.size, 1)
        print(
            f"  [save] sparse tracks {trajs.shape} N={trajs.shape[1]}"
            f" vis%={vis_pct:.1f} -> {out_tracks_path}"
        )

    # Optional dense per-pixel tracks: every pixel of the query frame becomes a
    # track, packaged as flow + visibility. Used by viz_tracks.py for the
    # "tracks for all pixels" overlay. Off by default (large files).
    if args.save_dense_tracks:
        flow_thw2_d, visibs_thw_d, confs_thw_d = _dense_tracks_from_flow(
            flows_e, visconf_e, args.track_conf_thr,
        )  # (T, H, W, 2) f16, (T, H, W) bool, (T, H, W) f16
        out_h_d = args.out_height or H
        out_w_d = args.out_width or W
        if (out_h_d, out_w_d) != (H, W):
            import cv2 as _cv2
            flow_thw2_d = _resize_flow(
                flow_thw2_d.astype(np.float32), out_h_d, out_w_d,
            ).astype(np.float16)

            def _resize_thw(arr, oh, ow, interp):
                return np.stack(
                    [_cv2.resize(a, (ow, oh), interpolation=interp) for a in arr], axis=0,
                )

            visibs_thw_d = _resize_thw(
                visibs_thw_d.astype(np.uint8), out_h_d, out_w_d, _cv2.INTER_NEAREST,
            ).astype(bool)
            confs_thw_d = _resize_thw(
                confs_thw_d.astype(np.float32), out_h_d, out_w_d, _cv2.INTER_LINEAR,
            ).astype(np.float16)
        out_dense_tracks_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dense_tracks_path,
            flow=flow_thw2_d,             # (T, H, W, 2) float16  pixel displacement
            visibs=visibs_thw_d,          # (T, H, W)    bool     vis (conf > thr)
            confs=confs_thw_d,            # (T, H, W)    float16  raw vis-conf
            image_size=np.array(
                [flow_thw2_d.shape[1], flow_thw2_d.shape[2]], dtype=np.int32,
            ),
            query_frame=np.int32(q),
            conf_thr=np.float32(args.track_conf_thr),
            fps=np.int32(fps),
        )
        vis_pct_d = 100.0 * visibs_thw_d.sum() / max(visibs_thw_d.size, 1)
        print(
            f"  [save] dense tracks flow={flow_thw2_d.shape} visibs={visibs_thw_d.shape}"
            f" vis%={vis_pct_d:.1f} -> {out_dense_tracks_path}"
        )

    # Optional dense-flow .npy (large; off by default). Used by legacy flow-
    # conditioning ablations; the world model's track head doesn't need it.
    if args.save_dense_flow:
        out_h = args.out_height or H
        out_w = args.out_width or W
        flow_thw2 = flows_e[0].permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
        if (out_h, out_w) != (H, W):
            flow_thw2 = _resize_flow(flow_thw2, out_h, out_w)
        out_flow_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_flow_path, flow_thw2)
        print(f"  [save] dense flow {flow_thw2.shape} -> {out_flow_path}")

    if args.no_viz:
        return

    # ---- viz: input | flow-color | sparse-track-overlay ----
    panels: List[np.ndarray] = []
    input_rgb = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    panels.append(input_rgb)
    panels.append(_flow2color_batch(flows_e, utils_improc))

    if not args.no_point_viz:
        trajs_viz = torch.from_numpy(trajs).to(rgbs.device)              # (T, N, 2)
        vis_viz = torch.from_numpy(visibs).to(rgbs.device)               # (T, N)
        colors = _get_2d_colors(queries_xy0, H, W, utils_improc)         # (N, 3)
        pts_rgb = _draw_pts_gpu(
            rgbs[0].clone(), trajs_viz, vis_viz, colors,
            radius=args.viz_radius, bkg_opacity=args.bkg_opacity,
        )
        panels.append(pts_rgb)

    viz = np.concatenate(panels, axis=2)  # (T, H, W*P, 3)
    _write_mp4(out_viz_path, viz, fps=fps)
    print(f"  [save] viz {viz.shape} -> {out_viz_path}")


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------

def _load_model(Net, utils_saveload, args):
    import torch
    window_len = args.window_len
    if args.tiny:
        model = Net(window_len, use_basicencoder=True, no_split=True)
    else:
        model = Net(window_len)

    if args.ckpt_init:
        if utils_saveload is not None and hasattr(utils_saveload, "load"):
            utils_saveload.load(
                None, args.ckpt_init, model,
                optimizer=None, scheduler=None, ignore_load=None,
                strict=True, verbose=False, weights_only=False,
            )
        else:
            sd = torch.load(args.ckpt_init, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            model.load_state_dict(sd, strict=True)
        print(f"[model] loaded ckpt {args.ckpt_init}")
    else:
        url = (
            "https://huggingface.co/aharley/alltracker/resolve/main/alltracker_tiny.pth"
            if args.tiny else
            "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
        )
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(sd["model"], strict=True)
        print(f"[model] loaded ckpt {url}")

    for p in model.parameters():
        p.requires_grad = False
    model.cuda().eval()
    return model


# ---------------------------------------------------------------------------
# metadata.csv update (dense_flow + tracks columns)
# ---------------------------------------------------------------------------

def _update_metadata(
    csv_path: Path,
    scene_to_flow_rel: dict,
    scene_to_tracks_rel: dict,
) -> None:
    """Add `tracks` (and optionally `dense_flow`) columns to the existing CSV.

    Scene name is taken from the basename (stem) of the `video` column, which
    `prepare_data_wan.py` writes as ``clips/<scene>.mp4`` (so the stem is the
    scene id we used to name the .npz / .npy files).
    """
    if not csv_path.is_file():
        print(f"[metadata] {csv_path} not found; skipping update")
        return
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    add_cols = []
    if scene_to_tracks_rel and "tracks" not in fieldnames:
        add_cols.append("tracks")
    if scene_to_flow_rel and "dense_flow" not in fieldnames:
        add_cols.append("dense_flow")
    fieldnames.extend(add_cols)
    for row in rows:
        scene = Path(row.get("video", "")).stem
        if scene in scene_to_tracks_rel:
            row["tracks"] = scene_to_tracks_rel[scene]
        if scene in scene_to_flow_rel:
            row["dense_flow"] = scene_to_flow_rel[scene]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    n_tracks = sum(1 for r in rows if r.get("tracks"))
    n_flow = sum(1 for r in rows if r.get("dense_flow"))
    print(f"[metadata] updated {csv_path}: tracks={n_tracks}/{len(rows)}"
          + (f", dense_flow={n_flow}/{len(rows)}" if scene_to_flow_rel else ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--alltracker_root", type=Path,
                   default=Path("/data/yjangir/sidegig/alltracker"),
                   help="Path to a working AllTracker checkout.")

    # Input selection. Either --droid_root (DROID layout) OR --clips_dir
    # (flat directory of <scene>.mp4). --droid_root is preferred: it targets
    # the REAL DROID camera MP4s, which is what AllTracker supervision needs.
    # The data_wan/clips/*.mp4 files are DrRobot RENDERINGS, not real DROID.
    p.add_argument("--droid_root", type=Path, default=DROID_ROOT,
                   help="Root of DROID scenes "
                        "(<root>/scene_*/recordings/MP4/<camera>.mp4). When "
                        "set, --clips_dir is ignored.")
    p.add_argument("--droid_camera", type=str, default="20103212",
                   help="DROID camera serial (filename stem inside "
                        "recordings/MP4/). Defaults to 20103212.")
    p.add_argument("--clips_dir", type=Path, default=CLIPS_DIR,
                   help="Fallback: flat directory of <scene>.mp4 to process "
                        "(used only when --no-droid_root is passed).")
    p.add_argument("--use_droid_root", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Whether to use --droid_root (default True). Pass "
                        "--no-use_droid_root to fall back to --clips_dir.")
    p.add_argument("--out_flow_dir", type=Path, default=OUT_FLOW_DIR)
    p.add_argument("--out_tracks_dir", type=Path, default=OUT_TRACKS_DIR)
    p.add_argument("--out_dense_tracks_dir", type=Path, default=OUT_DENSE_TRACKS_DIR)
    p.add_argument("--out_viz_dir", type=Path, default=OUT_VIZ_DIR)
    p.add_argument("--metadata_csv", type=Path, default=METADATA_CSV)
    p.add_argument("--update_metadata", action="store_true")

    # Output toggles.
    p.add_argument("--save_tracks", action=argparse.BooleanOptionalAction, default=True,
                   help="Save sparse point-track .npz (default: True). This is "
                        "what the world model's auxiliary head consumes.")
    p.add_argument("--save_dense_tracks", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Save dense per-pixel tracks .npz (flow + visibs + confs) "
                        "to --out_dense_tracks_dir. Off by default; large files.")
    p.add_argument("--save_dense_flow", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Also save dense (T,H,W,2) flow .npy (~500 MB / scene; "
                        "default: False). Used only by legacy flow ablations.")
    # Sparse query-grid resolution: defaults give 32 * 20 = 640 query points,
    # matching prior outputs. Total grid is grid_x * grid_y points.
    p.add_argument("--track_grid_x", type=int, default=32,
                   help="Number of query points along the x (width) axis.")
    p.add_argument("--track_grid_y", type=int, default=20,
                   help="Number of query points along the y (height) axis.")

    # AllTracker inference hyperparams.
    p.add_argument("--ckpt_init", type=str, default="",
                   help="Local AllTracker checkpoint (default: auto-download from HF).")
    p.add_argument("--tiny", action="store_true", help="Use the tiny AllTracker variant.")
    p.add_argument("--window_len", type=int, default=16)
    p.add_argument("--image_size", type=int, default=1024,
                   help="Max side (H or W) AllTracker sees. Larger = more VRAM.")
    p.add_argument("--inference_iters", type=int, default=4)
    p.add_argument("--query_frame", type=int, default=0,
                   help="Reference frame; flow + tracks are measured w.r.t. this frame.")
    p.add_argument("--max_frames", type=int, default=0,
                   help="Optional per-clip frame cap (0 = no cap).")
    p.add_argument("--max_scenes", type=int, default=0,
                   help="Optional scene cap (0 = process all).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing outputs instead of skipping.")

    # Flow-resizing.
    p.add_argument("--out_height", type=int, default=None)
    p.add_argument("--out_width", type=int, default=None)

    # Dense-track visibility threshold.
    p.add_argument("--track_conf_thr", type=float, default=0.1,
                   help="Visibility-confidence threshold used to binarize the "
                        "saved per-pixel `visibs` mask.")

    # Visualization (only affects the overlay panel; saved tracks are always dense).
    p.add_argument("--no_viz", action="store_true", help="Do not write viz MP4.")
    p.add_argument("--no_point_viz", action="store_true",
                   help="Skip the point-track panel; keep only flow-color.")
    p.add_argument("--viz_stride", type=int, default=16,
                   help="Pixel stride between points drawn in the track-overlay "
                        "panel of the viz MP4 (saved data is always dense).")
    p.add_argument("--viz_radius", type=int, default=2)
    p.add_argument("--bkg_opacity", type=float, default=0.5)

    args = p.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        args.max_frames = None
    if (not args.save_dense_flow and not args.save_tracks
            and not args.save_dense_tracks and args.no_viz):
        raise SystemExit(
            "nothing to save: pass at least one of --save_tracks, "
            "--save_dense_tracks, --save_dense_flow, or omit --no_viz."
        )

    # Build the (scene_name, mp4_path) work list.
    scene_clips: List[Tuple[str, Path]]
    if args.use_droid_root:
        print(f"[input] DROID: root={args.droid_root}  camera={args.droid_camera}.mp4")
        scene_clips = _enumerate_droid_clips(args.droid_root, args.droid_camera)
    else:
        print(f"[input] flat clips dir: {args.clips_dir}")
        mp4s = sorted(args.clips_dir.glob("*.mp4"))
        if not mp4s:
            raise SystemExit(f"no .mp4 clips found under {args.clips_dir}")
        scene_clips = [(mp.stem, mp) for mp in mp4s]
    if args.max_scenes > 0:
        scene_clips = scene_clips[: args.max_scenes]
    print(
        f"[plan] {len(scene_clips)} clips | save_tracks={args.save_tracks} "
        f"save_dense_flow={args.save_dense_flow} "
        f"grid={args.track_grid_y}x{args.track_grid_x} "
        f"(N={args.track_grid_x * args.track_grid_y} query points)"
    )

    import torch
    torch.set_grad_enabled(False)

    Net, utils_improc, utils_basic, utils_saveload = _import_alltracker(args.alltracker_root)
    model = _load_model(Net, utils_saveload, args)

    scene_to_flow_rel: dict = {}
    scene_to_tracks_rel: dict = {}
    ok, skipped, failed = 0, 0, 0
    for i, (scene, mp4) in enumerate(scene_clips):
        out_flow = args.out_flow_dir / f"{scene}.npy"
        out_tracks = args.out_tracks_dir / f"{scene}.npz"
        out_dense_tracks = args.out_dense_tracks_dir / f"{scene}.npz"
        out_viz = args.out_viz_dir / f"{scene}.mp4"
        print(f"\n=== [{i + 1}/{len(scene_clips)}] {scene}  <- {mp4} ===")

        have_flow = (not args.save_dense_flow) or out_flow.is_file()
        have_tracks = (not args.save_tracks) or out_tracks.is_file()
        have_dense_tracks = (not args.save_dense_tracks) or out_dense_tracks.is_file()
        have_viz = args.no_viz or out_viz.is_file()
        if (have_flow and have_tracks and have_dense_tracks and have_viz
                and not args.force):
            print("  [skip] outputs already exist (use --force to regenerate)")
            if args.save_dense_flow:
                scene_to_flow_rel[scene] = os.path.relpath(out_flow, DATA_WAN)
            if args.save_tracks:
                scene_to_tracks_rel[scene] = os.path.relpath(out_tracks, DATA_WAN)
            skipped += 1
            continue
        try:
            process_clip(
                model, mp4, out_flow, out_tracks, out_dense_tracks, out_viz,
                args, utils_improc, utils_basic,
            )
            if args.save_dense_flow:
                scene_to_flow_rel[scene] = os.path.relpath(out_flow, DATA_WAN)
            if args.save_tracks:
                scene_to_tracks_rel[scene] = os.path.relpath(out_tracks, DATA_WAN)
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {scene}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n[done] ok={ok}  skipped={skipped}  failed={failed}")

    if args.update_metadata:
        _update_metadata(args.metadata_csv, scene_to_flow_rel, scene_to_tracks_rel)


if __name__ == "__main__":
    main()
