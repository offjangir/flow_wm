#!/usr/bin/env python3
"""
Master data-generation pipeline for the WAN Video World Model.

Walks all DROID episode directories under --data_root, runs DrRobot rendering
and AllTracker 2D point-track extraction in parallel across GPUs, and writes
a train_metadata.csv manifest.

Robot-type → DrRobot model path is configured via --drrobot_config (JSON):
  {
    "panda":    "/data/.../checkpoints/drrobot_panda",
    "humanoid": "/data/.../checkpoints/drrobot_humanoid",
    "widowx":   "/data/.../checkpoints/drrobot_widowx"
  }

Architecture
------------
One persistent worker process is spawned per GPU.  Each worker:
  1. Loads AllTracker once from --alltracker_root (or HuggingFace).
  2. For each assigned episode, in order:
     a. Render  – calls render_drrobot_trajectory() in-process.
     b. Track   – calls AllTracker process_clip() in-process.
  3. Returns a result dict that is written into train_metadata.csv.

DrRobot is re-initialized per episode (not cached) so that different robot
types and model paths work correctly without import-cache conflicts.

Outputs
-------
  <out_dir>/renders/<scene>/drrobot_render.mp4
  <out_dir>/alltracker_tracks/<scene>.npz
  <out_dir>/alltracker_viz/<scene>.mp4          (unless --no_viz)
  <out_dir>/train_metadata.csv

Usage
-----
  export MUJOCO_GL=egl
  python scripts/scale_data_generation.py \\
      --data_root /data/yjangir/sidegig/wm/data/droid_10_demos \\
      --out_dir   /data/yjangir/sidegig/wm/data_wan \\
      --drrobot_config configs/drrobot_models.json \\
      --gpus 0 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─── Project paths ────────────────────────────────────────────────────────────

SIDEGIG = Path(__file__).resolve().parent.parent  # .../wm/

# ─── Demo discovery ───────────────────────────────────────────────────────────

_CAMERA_ROLE_TO_META_FIELD = {
    "wrist_left":  "wrist_cam_serial",
    "ext1_left":   "ext1_cam_serial",
    "ext1_right":  "ext1_cam_serial",
    "ext2_left":   "ext2_cam_serial",
    "ext2_right":  "ext2_cam_serial",
}


def _resolve_camera_serial(scene_dir: Path, meta: Dict[str, Any], camera_role: str, fallback: str) -> str:
    """Resolve the per-scene camera serial for the requested role.

    DROID camera serials vary across episodes/labs, so a fixed --droid_camera
    serial fails on most scenes. Order:
      1. metadata field for the role (e.g. ext1_cam_serial)
      2. any single MP4 present under recordings/MP4 (last-resort guess)
      3. the user-supplied fallback
    """
    field = _CAMERA_ROLE_TO_META_FIELD.get(camera_role.lower(), "")
    serial = str(meta.get(field, "")).strip() if field else ""
    if serial and (scene_dir / "recordings" / "MP4" / f"{serial}.mp4").is_file():
        return serial
    mp4_dir = scene_dir / "recordings" / "MP4"
    if mp4_dir.is_dir():
        mp4s = sorted(mp4_dir.glob("*.mp4"))
        if len(mp4s) == 1:
            return mp4s[0].stem
        if serial:
            for m in mp4s:
                if m.stem == serial:
                    return serial
    return fallback


def discover_demos(
    data_root: Path,
    droid_camera: str,
    camera_role: str,
    drrobot_config: Dict[str, str],
    default_robot_type: str,
    default_prompt: str,
) -> List[Dict[str, Any]]:
    """Return a list of per-episode job dicts.

    Each dict contains all information a worker needs: paths, robot type,
    model path, and prompt — all as plain strings so they cross process
    boundaries without pickling issues.

    The MP4 path is resolved per-scene from metadata (camera serials vary
    across DROID episodes/labs), with --droid_camera as a last-resort fallback.
    """
    demos: List[Dict[str, Any]] = []
    n_skipped_no_traj = 0
    n_skipped_no_mp4  = 0
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue

        traj = scene_dir / "trajectory.h5"
        if not traj.is_file():
            logging.warning("[discover] skip %s – no trajectory.h5", scene_dir.name)
            n_skipped_no_traj += 1
            continue

        meta = _load_metadata(scene_dir)
        serial = _resolve_camera_serial(scene_dir, meta, camera_role, droid_camera)
        mp4 = scene_dir / "recordings" / "MP4" / f"{serial}.mp4"
        if not mp4.is_file():
            logging.warning(
                "[discover] skip %s – no MP4 for role=%s (resolved serial=%s)",
                scene_dir.name, camera_role, serial,
            )
            n_skipped_no_mp4 += 1
            continue

        robot_type = _detect_robot_type(meta, default_robot_type)
        drrobot_model_path = drrobot_config.get(robot_type, "")
        if not drrobot_model_path:
            logging.warning(
                "[discover] %s: robot_type=%r has no DrRobot model in config; render will be skipped",
                scene_dir.name, robot_type,
            )

        prompt = _extract_prompt(meta, default_prompt)

        demos.append({
            "scene_name":        scene_dir.name,
            "trajectory_dir":    str(scene_dir),
            "real_mp4":          str(mp4),
            "robot_type":        robot_type,
            "drrobot_model_path": drrobot_model_path,
            "prompt":            prompt,
        })

    if not demos:
        raise SystemExit(f"No valid episodes found under {data_root}")

    logging.info(
        "[discover] found %d episodes (skipped: no_traj=%d, no_mp4=%d)",
        len(demos), n_skipped_no_traj, n_skipped_no_mp4,
    )
    return demos


def _load_metadata(scene_dir: Path) -> Dict[str, Any]:
    jsons = list(scene_dir.glob("*.json"))
    if not jsons:
        return {}
    with open(jsons[0], encoding="utf-8") as f:
        return json.load(f)


def _detect_robot_type(meta: Dict[str, Any], default: str) -> str:
    """Infer robot type from metadata; fall back to *default*."""
    if "robot_type" in meta:
        return str(meta["robot_type"]).lower()
    lab = meta.get("lab", "").lower()
    if "humanoid" in lab:
        return "humanoid"
    if "widowx" in lab or "widow" in lab:
        return "widowx"
    # DROID episodes are almost always Franka Panda.
    return default


def _extract_prompt(meta: Dict[str, Any], default: str) -> str:
    """Return first non-empty line of current_task, else default."""
    raw = meta.get("current_task", "") or meta.get("task", "") or default
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return default


# ─── Per-GPU worker (persistent process) ──────────────────────────────────────

# Module-level globals are per-process; workers populate them in _worker_init.
_WORKER_GPU_ID: int = -1
_ALLTRACKER_MODEL = None
_ALLTRACKER_UTILS_IMPROC = None
_ALLTRACKER_UTILS_BASIC = None
_WORKER_CFG: Dict[str, Any] = {}
# DrRobot model cache: keyed by (robot_type, model_path) so heterogeneous
# datasets still work, but for a homogenous Panda dataset we only load once.
_DRROBOT_MODELS: Dict[tuple, tuple] = {}


def _worker_init(gpu_queue: "mp.Queue", cfg: Dict[str, Any]) -> None:
    """Called once per worker process.

    Assigns a GPU, loads AllTracker, and pre-loads the DrRobot model(s)
    listed in ``cfg['drrobot_preload']`` so each worker pays the heavy
    Gaussian-checkpoint load cost exactly once.
    """
    global _WORKER_GPU_ID, _ALLTRACKER_MODEL
    global _ALLTRACKER_UTILS_IMPROC, _ALLTRACKER_UTILS_BASIC, _WORKER_CFG

    _WORKER_GPU_ID = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_WORKER_GPU_ID)
    _WORKER_CFG = cfg

    logging.basicConfig(
        level=logging.INFO,
        format=f"[GPU{_WORKER_GPU_ID}] %(levelname)s %(message)s",
        force=True,
    )
    log = logging.getLogger()

    if not cfg.get("skip_tracks"):
        log.info("Loading AllTracker …")
        _ALLTRACKER_MODEL, _ALLTRACKER_UTILS_IMPROC, _ALLTRACKER_UTILS_BASIC = (
            _load_alltracker(cfg)
        )
        log.info("AllTracker ready.")
    else:
        log.info("Tracking disabled (--skip_tracks).")

    if not cfg.get("skip_render"):
        for robot_type, model_path in cfg.get("drrobot_preload", []):
            log.info("Loading DrRobot model: %s (%s) …", robot_type, model_path)
            _get_drrobot_model(robot_type, model_path, cfg, log)


def _get_drrobot_model(robot_type: str, model_path: str, cfg: Dict[str, Any], log):
    """Lazy-load + cache (gaussians, bg_color, base_camera) for a DrRobot run."""
    key = (robot_type, model_path)
    if key in _DRROBOT_MODELS:
        return _DRROBOT_MODELS[key]

    drrobot_root = Path(cfg["drrobot_root"]).resolve()
    drrobot_str = str(drrobot_root)
    if drrobot_str not in sys.path:
        sys.path.insert(0, drrobot_str)

    # Module name `render_droid_scenes` is a convenient handle.
    import importlib.util
    rds_path = drrobot_root / "render_droid_scenes.py"
    spec = importlib.util.spec_from_file_location("render_droid_scenes", rds_path)
    rds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rds)  # type: ignore[union-attr]

    # safe_state once per worker is enough; idempotent if called per-model.
    rds.safe_state(True)
    background = cfg.get("background", "white")
    gaussians, bg_color, sample_cameras = rds.load_model(model_path, background=background)
    base_camera = sample_cameras[0]
    _DRROBOT_MODELS[key] = (rds, gaussians, bg_color, base_camera)
    log.info(
        "  DrRobot loaded: chain_dof=%d  bg=%s  sample_cam=%dx%d",
        len(gaussians.chain.get_joint_limits()[0]),
        background, base_camera.image_width, base_camera.image_height,
    )
    return _DRROBOT_MODELS[key]


def _load_alltracker(cfg: Dict[str, Any]):
    """Load AllTracker model; returns (model, utils_improc, utils_basic)."""
    import torch

    alltracker_root = Path(cfg["alltracker_root"])
    if not (alltracker_root / "nets" / "alltracker.py").is_file():
        raise SystemExit(
            f"AllTracker repo not found at {alltracker_root}\n"
            "Clone it first:\n"
            f"  git clone https://github.com/aharley/alltracker.git {alltracker_root}"
        )

    root_str = str(alltracker_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from nets.alltracker import Net  # type: ignore

    def _try(name):
        try:
            return __import__(name, fromlist=["*"])
        except Exception as exc:
            logging.warning("Could not import alltracker %s: %s", name, exc)
            return None

    utils_improc = _try("utils.improc")
    utils_basic  = _try("utils.basic")

    window_len = cfg.get("window_len", 16)
    model = Net(window_len)

    ckpt_path = cfg.get("alltracker_ckpt", "")
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        model.load_state_dict(sd, strict=True)
        logging.info("AllTracker: loaded local ckpt %s", ckpt_path)
    else:
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(sd["model"], strict=True)
        logging.info("AllTracker: loaded HuggingFace checkpoint.")

    torch.set_grad_enabled(False)
    model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, utils_improc, utils_basic


# ─── Per-episode processing ───────────────────────────────────────────────────

def _process_episode(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function: render + track one episode.  Returns result dict."""
    cfg = _WORKER_CFG
    scene      = job["scene_name"]
    traj_dir   = Path(job["trajectory_dir"])
    real_mp4   = Path(job["real_mp4"])
    robot_type = job["robot_type"]
    model_path = job["drrobot_model_path"]

    out_dir       = Path(cfg["out_dir"])
    render_dir    = out_dir / "renders" / scene
    render_mp4    = render_dir / "drrobot_render.mp4"
    tracks_path   = out_dir / "alltracker_tracks" / f"{scene}.npz"
    dense_path    = out_dir / "alltracker_dense_tracks" / f"{scene}.npz"
    viz_path      = out_dir / "alltracker_viz" / f"{scene}.mp4"
    force         = cfg.get("force", False)
    save_dense    = bool(cfg.get("save_dense_tracks", False))

    log = logging.getLogger()
    log.info("=== %s [robot=%s] ===", scene, robot_type)

    # ── Render ────────────────────────────────────────────────────────────
    if not cfg.get("skip_render") and model_path:
        if force or not render_mp4.is_file():
            log.info("  Rendering …")
            _run_render(traj_dir, render_dir, robot_type, model_path, cfg, log)
        else:
            log.info("  [skip render] %s exists", render_mp4.name)
    elif not model_path and not cfg.get("skip_render"):
        log.warning("  [skip render] no DrRobot model path for robot_type=%r", robot_type)

    # ── Track ─────────────────────────────────────────────────────────────
    if not cfg.get("skip_tracks"):
        sparse_ok = tracks_path.is_file()
        dense_ok  = (not save_dense) or dense_path.is_file()
        if force or not (sparse_ok and dense_ok):
            log.info("  Tracking%s …", "  (+dense)" if save_dense else "")
            _run_tracks(
                real_mp4, tracks_path,
                dense_path if save_dense else None,
                viz_path, cfg, log,
            )
        else:
            log.info("  [skip tracks] %s exists%s",
                     tracks_path.name,
                     " (+dense)" if save_dense and dense_ok else "")

    return {
        "scene_name":   scene,
        "video":        str(real_mp4),
        "render":       str(render_mp4) if render_mp4.is_file() else "",
        "tracks":       str(tracks_path) if tracks_path.is_file() else "",
        "dense_tracks": str(dense_path) if (save_dense and dense_path.is_file()) else "",
        "prompt":       job["prompt"],
        "robot_type":   robot_type,
    }


_CAMERA_ROLE_TO_KEY = {
    "wrist_left":  ("wrist", "left"),
    "ext1_left":   ("ext1",  "left"),
    "ext1_right":  ("ext1",  "right"),
    "ext2_left":   ("ext2",  "left"),
    "ext2_right":  ("ext2",  "right"),
}


def _run_render(
    traj_dir: Path,
    work_dir: Path,
    robot_type: str,
    drrobot_model_path: str,
    cfg: Dict[str, Any],
    log: logging.Logger,
) -> None:
    """Render one DROID episode using ``drrobot/render_droid_scenes.py``.

    Uses per-scene ZED intrinsics + the real DROID 1280×720 image size and a
    white background, matching the real ext1 camera framing. The DrRobot
    Gaussian model is loaded once per worker and cached.
    """
    import copy

    rds, gaussians, bg_color, base_camera = _get_drrobot_model(
        robot_type, drrobot_model_path, cfg, log,
    )

    camera_role = cfg.get("camera_role", "ext1_left")
    camera_key, camera_stereo = _CAMERA_ROLE_TO_KEY.get(camera_role, ("ext1", "left"))

    width  = cfg.get("render_width", 1280)
    height = cfg.get("render_height", 720)

    # Per-scene ZED intrinsics (online fetch + hardcoded table fallback).
    zed = rds.get_scene_camera_intrinsics(str(traj_dir), camera_key)
    log.info("  ZED intrinsics: fx=%.2f fy=%.2f", zed["fx"], zed["fy"])

    cam = copy.deepcopy(base_camera)
    rds.override_camera_intrinsics(
        cam, width, height,
        fov_x_deg=cfg.get("render_fov", None),
        fx=zed["fx"], fy=zed["fy"],
    )

    # FPS: match real recording (DROID is typically 60 Hz) unless overridden.
    fps_cfg = cfg.get("fps", "match")
    if isinstance(fps_cfg, str) and fps_cfg.lower() in ("match", "auto"):
        fps = rds.recording_fps_for_camera(str(traj_dir), camera_key) or 15.0
    else:
        fps = float(fps_cfg)
    log.info("  Render: %dx%d  fps=%.4g  bg=%s", width, height, fps,
             cfg.get("background", "white"))

    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "drrobot_render.mp4"

    rds.render_scene_video(
        scene_dir=str(traj_dir),
        output_path=str(output_path),
        gaussians=gaussians,
        bg_color=bg_color,
        base_camera=cam,
        fps=fps,
        camera_key=camera_key,
        use_trajectory_camera=cfg.get("trajectory_camera", False),
        camera_stereo=camera_stereo,
        orbit_params=None,
        gl_flip=cfg.get("gl_flip", False),
    )
    log.info("  Render → %s", output_path)


def _run_tracks(
    mp4_path: Path,
    out_tracks_path: Path,
    out_dense_path: Optional[Path],
    out_viz_path: Path,
    cfg: Dict[str, Any],
    log: logging.Logger,
) -> None:
    """Run AllTracker on *mp4_path*, save sparse + (optionally) dense tracks.

    Memory-safety: frames are uniformly subsampled across the whole episode
    (not first-N truncated) so the tracks span the same duration as the render.
    CUDA cache is cleared at the end; OOM on a single scene must not poison
    the worker for subsequent scenes.
    """
    import numpy as np
    import torch
    import time
    import gc

    # Re-import the helper functions directly from extract_alltracker.
    scripts_dir = str((SIDEGIG / "scripts").resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from extract_alltracker import (  # type: ignore
        _read_mp4_frames,
        _resize_keep_ar,
        _sparse_query_grid,
        _sparse_tracks_from_flow,
        _dense_tracks_from_flow,
        _forward_alltracker,
        _flow2color_batch,
        _get_2d_colors,
        _draw_pts_gpu,
        _write_mp4,
    )

    model       = _ALLTRACKER_MODEL
    utils_improc = _ALLTRACKER_UTILS_IMPROC

    max_frames     = cfg.get("max_frames") or None
    image_size     = cfg.get("image_size", 768)
    query_frame    = cfg.get("query_frame", 0)
    inference_iters = cfg.get("inference_iters", 4)
    grid_x         = cfg.get("track_grid_x", 32)
    grid_y         = cfg.get("track_grid_y", 20)
    conf_thr       = cfg.get("track_conf_thr", 0.1)
    no_viz         = cfg.get("no_viz", False)
    save_dense     = bool(cfg.get("save_dense_tracks", False))

    # Read all frames, then uniformly subsample to max_frames across the whole
    # episode duration so tracks span the same time as the render.
    frames, fps = _read_mp4_frames(mp4_path, None)
    T_full = len(frames)
    if max_frames and T_full > max_frames:
        idxs = np.linspace(0, T_full - 1, max_frames, dtype=np.int64)
        frames = [frames[i] for i in idxs]

    frames, (H, W) = _resize_keep_ar(frames, image_size)
    T = len(frames)
    q = max(0, min(query_frame, T - 1))

    try:
        rgbs = torch.stack(
            [torch.from_numpy(f).permute(2, 0, 1) for f in frames], dim=0
        ).unsqueeze(0).float().cuda()

        t0 = time.time()
        flows_e, visconf_e = _forward_alltracker(model, rgbs, q, inference_iters)
        dt = time.time() - t0
        log.info("  AllTracker: %d frames (of %d) @ %dx%d in %.1fs",
                 T, T_full, H, W, dt)

        # Sparse tracks (always, cheap).
        queries_xy0 = _sparse_query_grid(H, W, grid_x, grid_y)
        trajs, visibs, confs = _sparse_tracks_from_flow(
            flows_e, visconf_e, queries_xy0, conf_thr,
        )
        out_tracks_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_tracks_path,
            trajs=trajs,
            visibs=visibs,
            confs=confs,
            queries_xy0=queries_xy0,
            image_size=np.array([H, W], dtype=np.int32),
            query_frame=np.int32(q),
            conf_thr=np.float32(conf_thr),
            fps=np.int32(fps),
            T_source=np.int32(T_full),
        )
        vis_pct = 100.0 * visibs.sum() / max(visibs.size, 1)
        log.info("  Sparse tracks %s  vis=%.1f%%  → %s",
                 trajs.shape, vis_pct, out_tracks_path)

        # Dense tracks (per-pixel flow + visibility, float16).
        if save_dense and out_dense_path is not None:
            flow_thw2_d, visibs_thw_d, confs_thw_d = _dense_tracks_from_flow(
                flows_e, visconf_e, conf_thr,
            )
            out_dense_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dense_path,
                flow_thw2=flow_thw2_d,
                visibs_thw=visibs_thw_d,
                confs_thw=confs_thw_d,
                image_size=np.array([H, W], dtype=np.int32),
                query_frame=np.int32(q),
                conf_thr=np.float32(conf_thr),
                fps=np.int32(fps),
                T_source=np.int32(T_full),
            )
            vis_pct_d = 100.0 * visibs_thw_d.sum() / max(visibs_thw_d.size, 1)
            log.info("  Dense tracks flow=%s vis=%.1f%% → %s",
                     flow_thw2_d.shape, vis_pct_d, out_dense_path)

        if not no_viz:
            panels = []
            input_rgb = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            panels.append(input_rgb)
            panels.append(_flow2color_batch(flows_e, utils_improc))
            trajs_t  = torch.from_numpy(trajs).to(rgbs.device)
            visibs_t = torch.from_numpy(visibs).to(rgbs.device)
            colors   = _get_2d_colors(queries_xy0, H, W, utils_improc)
            pts_rgb  = _draw_pts_gpu(rgbs[0].clone(), trajs_t, visibs_t, colors)
            panels.append(pts_rgb)
            viz = np.concatenate(panels, axis=2)
            _write_mp4(out_viz_path, viz, fps=fps)
            log.info("  Viz → %s", out_viz_path)
    finally:
        # Always drop CUDA tensors before returning so the next scene starts
        # with an empty cache — protects against single-scene OOM cascading.
        for v in ("flows_e", "visconf_e", "rgbs"):
            if v in locals():
                del locals()[v]
        gc.collect()
        torch.cuda.empty_cache()


# ─── Worker pool entry point ───────────────────────────────────────────────────

def _episode_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Pool map target; wraps _process_episode with per-episode error capture."""
    try:
        return _process_episode(job)
    except Exception:
        scene = job.get("scene_name", "?")
        tb = traceback.format_exc()
        logging.getLogger().error("[FAIL] %s\n%s", scene, tb)
        return {
            "scene_name":   scene,
            "video":        job.get("real_mp4", ""),
            "render":       "",
            "tracks":       "",
            "dense_tracks": "",
            "prompt":       job.get("prompt", ""),
            "robot_type":   job.get("robot_type", ""),
            "error":        tb.splitlines()[-1],
        }


# ─── Manifest writer ──────────────────────────────────────────────────────────

def write_manifest(results: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video", "render", "tracks", "dense_tracks", "prompt", "robot_type"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    n_render = sum(1 for r in results if r.get("render"))
    n_tracks = sum(1 for r in results if r.get("tracks"))
    n_dense  = sum(1 for r in results if r.get("dense_tracks"))
    logging.info(
        "Manifest → %s  (%d rows, render=%d, tracks=%d, dense=%d)",
        csv_path, len(results), n_render, n_tracks, n_dense,
    )


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    p.add_argument("--data_root", type=Path, default=SIDEGIG / "data" / "droid_10_demos",
                   help="Root containing scene_* episode directories.")
    p.add_argument("--out_dir", type=Path, default=SIDEGIG / "data_wan",
                   help="Output root (renders/, alltracker_tracks/, train_metadata.csv).")
    p.add_argument("--drrobot_config", type=Path,
                   default=SIDEGIG / "configs" / "drrobot_models.json",
                   help="JSON mapping robot_type → DrRobot model path.")
    p.add_argument("--drrobot_root", type=Path, default=SIDEGIG / "drrobot",
                   help="DrRobot repository root.")
    p.add_argument("--alltracker_root", type=Path,
                   default=Path("/data/yjangir/sidegig/alltracker"),
                   help="AllTracker repository root.")
    p.add_argument("--alltracker_ckpt", type=str, default="",
                   help="Local AllTracker checkpoint (default: auto-download from HF).")

    # Parallelism
    p.add_argument("--gpus", type=int, nargs="+", default=[0],
                   help="GPU IDs to use (one worker per GPU).")

    # Camera
    p.add_argument("--camera_role", type=str, default="ext1_left",
                   choices=["wrist_left", "ext1_left", "ext1_right", "ext2_left", "ext2_right"],
                   help="Which DROID camera to use for DrRobot rendering.")
    p.add_argument("--droid_camera", type=str, default="20103212",
                   help="Fallback camera serial used only when the per-scene metadata "
                        "does not specify one for --camera_role.")

    # Rendering (uses drrobot/render_droid_scenes.py per-scene)
    p.add_argument("--fps", type=str, default="match",
                   help="'match' = ffprobe the real DROID MP4 for FPS (typical 60); "
                        "or pass a positive number for a fixed-rate render (e.g. 15).")
    p.add_argument("--render_width",  type=int, default=1280,
                   help="Render width (DROID native = 1280).")
    p.add_argument("--render_height", type=int, default=720,
                   help="Render height (DROID native = 720).")
    p.add_argument("--render_fov", type=float, default=None,
                   help="Override horizontal FOV in degrees (else uses ZED intrinsics).")
    p.add_argument("--background", type=str, default="black",
                   choices=["white", "black"],
                   help="Gaussian rasterizer clear color behind the robot. "
                        "Black matches the existing reference renders the model "
                        "trained on (white-ish robot stands out cleanly).")
    p.add_argument("--trajectory_camera", action="store_true",
                   help="Use per-frame extrinsics from trajectory.h5 instead of static metadata.")
    p.add_argument("--gl_flip", action="store_true",
                   help="Apply 180-deg X-flip when interpreting DROID 6-DoF extrinsics.")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="(Reserved) currently ignored — render_droid_scenes.py renders all frames.")
    p.add_argument("--max_frames", type=int, default=0,
                   help="(Reserved) currently ignored.")

    # AllTracker
    p.add_argument("--image_size", type=int, default=768,
                   help="Max side (H or W) AllTracker resizes to. 768 keeps "
                        "VRAM in check for dense-track export at T~=200.")
    p.add_argument("--tracker_max_frames", type=int, default=200,
                   help="Uniformly subsample the real video to at most this "
                        "many frames before AllTracker. 0 = no cap. Tracks "
                        "still span the full episode duration.")
    p.add_argument("--inference_iters", type=int, default=4)
    p.add_argument("--window_len", type=int, default=16)
    p.add_argument("--query_frame", type=int, default=0)
    p.add_argument("--track_grid_x", type=int, default=32,
                   help="Sparse query points along x (width).")
    p.add_argument("--track_grid_y", type=int, default=20,
                   help="Sparse query points along y (height).")
    p.add_argument("--track_conf_thr", type=float, default=0.1)
    p.add_argument("--save_dense_tracks", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Also write per-pixel dense tracks (flow + visibility, "
                        "float16) to <out_dir>/alltracker_dense_tracks/<scene>.npz.")
    p.add_argument("--no_viz", action="store_true",
                   help="Skip AllTracker visualization MP4.")

    # Robot / prompt defaults
    p.add_argument("--default_robot_type", type=str, default="panda",
                   help="Robot type used when metadata provides no hint.")
    p.add_argument("--default_prompt", type=str,
                   default="robot arm manipulation",
                   help="Prompt used when episode metadata has no task description.")

    # Control flags
    p.add_argument("--skip_render", action="store_true",
                   help="Skip DrRobot rendering (expect renders to already exist).")
    p.add_argument("--skip_tracks", action="store_true",
                   help="Skip AllTracker tracking.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing render/tracks outputs.")

    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[main] %(levelname)s %(message)s")
    args = parse_args()

    # ── Load DrRobot config ────────────────────────────────────────────────
    drrobot_config: Dict[str, str] = {}
    if args.drrobot_config.is_file():
        with open(args.drrobot_config, encoding="utf-8") as f:
            drrobot_config = json.load(f)
        logging.info("DrRobot config: %s", drrobot_config)
    else:
        logging.warning(
            "DrRobot config not found at %s – rendering will be skipped for all episodes.",
            args.drrobot_config,
        )

    # ── Discover episodes ──────────────────────────────────────────────────
    demos = discover_demos(
        args.data_root,
        droid_camera=args.droid_camera,
        camera_role=args.camera_role,
        drrobot_config=drrobot_config,
        default_robot_type=args.default_robot_type,
        default_prompt=args.default_prompt,
    )

    # Pre-load list: every (robot_type → model_path) actually used by the demos.
    needed_robot_types = {d["robot_type"] for d in demos if d.get("drrobot_model_path")}
    drrobot_preload = [
        (rt, drrobot_config[rt]) for rt in sorted(needed_robot_types)
        if rt in drrobot_config and drrobot_config[rt]
    ]

    # ── Build worker config dict (plain types, pickle-safe) ───────────────
    cfg: Dict[str, Any] = {
        "out_dir":          str(args.out_dir),
        "drrobot_root":     str(args.drrobot_root),
        "drrobot_preload":  drrobot_preload,
        "alltracker_root":  str(args.alltracker_root),
        "alltracker_ckpt":  args.alltracker_ckpt,
        "camera_role":      args.camera_role,
        "fps":              args.fps,
        "render_width":     args.render_width,
        "render_height":    args.render_height,
        "render_fov":       args.render_fov,
        "background":       args.background,
        "trajectory_camera": args.trajectory_camera,
        "gl_flip":          args.gl_flip,
        "frame_stride":     args.frame_stride,
        "max_frames":       args.tracker_max_frames if args.tracker_max_frames > 0 else 0,
        "image_size":       args.image_size,
        "inference_iters":  args.inference_iters,
        "window_len":       args.window_len,
        "query_frame":      args.query_frame,
        "track_grid_x":     args.track_grid_x,
        "track_grid_y":     args.track_grid_y,
        "track_conf_thr":   args.track_conf_thr,
        "save_dense_tracks": args.save_dense_tracks,
        "no_viz":           args.no_viz,
        "skip_render":      args.skip_render,
        "skip_tracks":      args.skip_tracks,
        "force":            args.force,
    }

    gpu_ids = args.gpus
    num_workers = len(gpu_ids)
    logging.info("Launching %d GPU worker(s) for %d episodes …", num_workers, len(demos))

    # ── Launch pool ────────────────────────────────────────────────────────
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    for gid in gpu_ids:
        gpu_queue.put(gid)

    with mp.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(gpu_queue, cfg),
    ) as pool:
        results = list(pool.imap(_episode_worker, demos, chunksize=1))

    # ── Write manifest ─────────────────────────────────────────────────────
    manifest_path = args.out_dir / "train_metadata.csv"
    write_manifest(results, manifest_path)

    ok     = sum(1 for r in results if not r.get("error"))
    failed = sum(1 for r in results if r.get("error"))
    logging.info("Done. ok=%d  failed=%d  manifest=%s", ok, failed, manifest_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
