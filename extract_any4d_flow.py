#!/usr/bin/env python3
"""
Extract dense 2D optical flow from Any4D's 3D scene flow predictions.

Given DrRobot-rendered frames and a foreground mask, runs Any4D inference and
projects the per-frame 3D scene flow to 2D pixel displacement relative to the
reference frame.  Output: ``(T, H, W, 2)`` float32 ``.npy`` suitable for the
world-model training CSV (``flow`` column).

Must be run with the Any4D directory as cwd (for hydra config resolution),
or with Any4D installed as an editable package.

Usage (from Any4D repo root):
  python /path/to/extract_any4d_flow.py \\
    --frames_dir /path/to/drrobot/frames \\
    --mask_path  /path/to/ref_binary_mask.png \\
    --checkpoint checkpoints/any4d_4v_combined.pth \\
    --output     /path/to/flow.npy \\
    --start_idx 0 --end_idx 40 --ref_idx 0 \\
    --target_height 360 --target_width 640
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model / config helpers (inlined from Any4D demo_inference.py so this script
# stays standalone without importing the scripts/ directory as a package).
# ---------------------------------------------------------------------------

def _init_hydra_config(config_path: str, overrides=None):
    import hydra

    # Resolve against current working directory (prepare_data_wan runs this with cwd=Any4D).
    # This avoids accidentally resolving against wm/configs when this script lives outside Any4D.
    config_abs = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_abs)
    config_name = os.path.basename(config_path).split(".")[0]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=config_name, overrides=overrides or [])
    return cfg


def _init_model(config: dict, device: torch.device):
    from any4d.models import init_model

    config_path = config["path"]
    overrides = config.get("config_overrides", [])
    model_args = _init_hydra_config(config_path, overrides=overrides)

    model = init_model(model_args.model.model_str, model_args.model.model_config)
    model.to(device)
    ckpt_path = config["checkpoint_path"]
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 3D scene flow → 2D optical flow projection
# ---------------------------------------------------------------------------

def _project_to_2d(pts_cam: torch.Tensor, fx: float, fy: float, cx: float, cy: float):
    """Project (H, W, 3) camera-frame points to (H, W, 2) pixel coords [u, v]."""
    z = pts_cam[..., 2:3].clamp(min=1e-6)
    u = fx * pts_cam[..., 0:1] / z + cx
    v = fy * pts_cam[..., 1:2] / z + cy
    return torch.cat([u, v], dim=-1)  # (H, W, 2)


def scene_flow_to_2d(
    pts3d_cam: torch.Tensor,
    scene_flow_world: torch.Tensor,
    R_w2c: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """
    Single-frame: 3D allo scene flow → 2D pixel displacement.

    pts3d_cam       (H, W, 3)  reference points in camera frame
    scene_flow_world(H, W, 3)  world-frame scene flow ref → t
    R_w2c           (3, 3)     world-to-camera rotation

    Returns (H, W, 2) float32 numpy array [du, dv].
    """
    H, W, _ = pts3d_cam.shape
    sf_cam = (scene_flow_world.reshape(-1, 3) @ R_w2c.T).reshape(H, W, 3)
    pts_moved = pts3d_cam + sf_cam

    pix_ref = _project_to_2d(pts3d_cam, fx, fy, cx, cy)
    pix_moved = _project_to_2d(pts_moved, fx, fy, cx, cy)

    return (pix_moved - pix_ref).numpy().astype(np.float32)


def resize_flow(flow: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize (T, H, W, 2) flow, scaling displacement magnitudes accordingly."""
    _, h_src, w_src, _ = flow.shape
    if h_src == target_h and w_src == target_w:
        return flow
    sx, sy = target_w / w_src, target_h / h_src
    x = torch.from_numpy(flow).permute(0, 3, 1, 2).float()  # (T, 2, H, W)
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    out = x.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)  # (T, H, W, 2)
    out[..., 0] *= sx
    out[..., 1] *= sy
    return out


def _to_float_scalar(x, name: str) -> float:
    """Convert tensor/ndarray/string-like scalar to float with robust fallbacks."""
    if torch.is_tensor(x):
        if x.numel() == 0:
            raise ValueError(f"{name} is an empty tensor")
        return float(x.detach().reshape(-1)[0].cpu().item())
    if isinstance(x, np.ndarray):
        if x.size == 0:
            raise ValueError(f"{name} is an empty ndarray")
        return float(np.asarray(x).reshape(-1)[0].item())
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return float(s)
        except ValueError:
            # Handle formats like "tensor(533.41, device='cuda:0')"
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if m:
                return float(m.group(0))
        raise TypeError(f"Cannot parse {name} from string: {x!r}")
    raise TypeError(f"Unsupported type for {name}: {type(x)} value={x!r}")


def _parse_intrinsics(K_obj) -> tuple[float, float, float, float]:
    """
    Parse intrinsics from Any4D helper output across versions.

    Supports:
    - dict with keys fx/fy/cx/cy
    - 3x3 intrinsic matrix (torch or numpy)
    """
    if isinstance(K_obj, dict):
        return (
            _to_float_scalar(K_obj["fx"], "fx"),
            _to_float_scalar(K_obj["fy"], "fy"),
            _to_float_scalar(K_obj["cx"], "cx"),
            _to_float_scalar(K_obj["cy"], "cy"),
        )

    if torch.is_tensor(K_obj):
        k = K_obj.detach().cpu().numpy()
    elif isinstance(K_obj, np.ndarray):
        k = K_obj
    else:
        raise TypeError(f"Unsupported intrinsics type: {type(K_obj)}")

    k = np.asarray(k)
    if k.ndim >= 2 and k.shape[-2:] == (3, 3):
        m = k.reshape(-1, 3, 3)[0]
        fx = float(m[0, 0])
        fy = float(m[1, 1])
        cx = float(m[0, 2])
        cy = float(m[1, 2])
        return fx, fy, cx, cy

    raise ValueError(f"Cannot parse intrinsics from shape {k.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_flow(
    frames_dir: str,
    checkpoint: str,
    output_path: str,
    mask_path: str | None = None,
    start_idx: int = 0,
    end_idx: int = 40,
    ref_idx: int = 0,
    target_height: int | None = None,
    target_width: int | None = None,
    chunk_size: int = 12,
) -> np.ndarray:
    """Run Any4D and return reference-anchored 2D flow (T, H, W, 2)."""
    from glob import glob
    from natsort import natsorted

    from any4d.utils.geometry import (
        quaternion_to_rotation_matrix,
        recover_pinhole_intrinsics_from_ray_directions,
    )
    from any4d.utils.image import load_images
    from any4d.utils.inference import loss_of_one_batch_multi_view
    from any4d.utils.moge_inference import load_moge_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "path": "configs/train.yaml",
        "config_overrides": [
            "machine=local",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            "model/task=images_only",
        ],
        "checkpoint_path": checkpoint,
        "trained_with_amp": True,
        "data_norm_type": "dinov2",
    }
    model = _init_model(config, device)
    moge_model = load_moge_model(device=device.type)

    image_paths = natsorted(glob(os.path.join(frames_dir, "*.jpg")))
    n = len(image_paths)
    if n == 0:
        raise ValueError(f"No .jpg images found in {frames_dir}")
    start_idx = max(0, min(start_idx, n - 1))
    end_idx = max(start_idx + 1, min(end_idx, n))
    ref_idx = max(0, min(ref_idx, n - 1))

    resolution = (518, 336)
    n_frames = end_idx - start_idx
    flow_all: np.ndarray | None = None
    H_a, W_a = 0, 0
    frame_indices = list(range(start_idx, end_idx))
    if chunk_size <= 0:
        chunk_size = len(frame_indices)

    # Reference frame must be first view for scene-flow keys to be consistent.
    ref_path = image_paths[ref_idx]
    geometry_ready = False

    for chunk_start in range(0, len(frame_indices), chunk_size):
        chunk_indices = frame_indices[chunk_start : chunk_start + chunk_size]
        image_list = [ref_path] + [image_paths[i] for i in chunk_indices]
        views = load_images(
            image_list,
            size=resolution,
            verbose=False,
            norm_type=config["data_norm_type"],
            patch_size=14,
            compute_moge_mask=True,
            moge_model=moge_model,
            binary_mask_path=mask_path,
        )

        pred = loss_of_one_batch_multi_view(views, model, None, device, use_amp=True)

        if not geometry_ready:
            cam_quats = pred["pred1"]["cam_quats"][0].cpu()
            R_c2w = quaternion_to_rotation_matrix(cam_quats)
            R_w2c = R_c2w.T

            pts3d_cam = pred["pred1"]["pts3d_cam"][0].cpu()  # (H_a, W_a, 3)
            ray_dirs = pred["pred1"]["ray_directions"][0].cpu()
            K_obj = recover_pinhole_intrinsics_from_ray_directions(ray_dirs)
            fx, fy, cx, cy = _parse_intrinsics(K_obj)
            H_a, W_a = pts3d_cam.shape[:2]
            flow_all = np.zeros((n_frames, H_a, W_a, 2), dtype=np.float32)
            geometry_ready = True

        for idx in range(1, len(views)):
            pred_key = f"pred{idx + 1}"
            if pred_key not in pred or "scene_flow" not in pred[pred_key]:
                continue
            sf_world = pred[pred_key]["scene_flow"][0].cpu()
            global_t = chunk_start + (idx - 1)
            if global_t < n_frames:
                flow_all[global_t] = scene_flow_to_2d(pts3d_cam, sf_world, R_w2c, fx, fy, cx, cy)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[extract_flow] processed chunk {chunk_start}:{chunk_start + len(chunk_indices)} / {len(frame_indices)}"
        )

    if flow_all is None:
        raise RuntimeError("Any4D returned no valid predictions for flow extraction.")

    th = target_height or H_a
    tw = target_width or W_a
    flow_all = resize_flow(flow_all, th, tw)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, flow_all)
    print(f"Saved flow {flow_all.shape} → {output_path}")
    return flow_all


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Output .npy path")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=40)
    parser.add_argument("--ref_idx", type=int, default=0)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=12,
        help="Number of target frames per Any4D forward pass (lower = less GPU memory).",
    )
    args = parser.parse_args()

    extract_flow(
        frames_dir=args.frames_dir,
        checkpoint=args.checkpoint,
        output_path=args.output,
        mask_path=args.mask_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        ref_idx=args.ref_idx,
        target_height=args.target_height,
        target_width=args.target_width,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
