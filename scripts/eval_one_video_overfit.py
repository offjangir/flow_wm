#!/usr/bin/env python3
"""
Single-sample eval + numeric overfit metrics for Wan I2V finetuning (FSDP checkpoints).

Use this to sanity-check that a small finetune (e.g. ``drop_render_conditioning`` +
last blocks on one CSV row) actually moves predictions toward the ground-truth
clip, without scanning the full dataset.

What it does for one metadata row (default: index 0):

  1. Loads GT camera frames and (unless ``--drop_render_conditioning``) the render clip.
  2. Builds ``RenderConditionedWanI2VPipeline`` and optionally loads ``render_conditioner.pt``.
  3. Runs sampling from the first GT frame (+ prompt + optional render).
  4. Writes a compact MP4 (GT | pred [, render]) and a JSON summary with MSE/MAE/PSNR vs GT.

Run from the ``wm`` repo root::

  PYTHONPATH=src python scripts/eval_one_video_overfit.py \\
    --model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \\
    --checkpoint_path ./checkpoints/.../epoch_N/render_conditioner.pt \\
    --metadata_csv ./data_wan/metadata_one_example.csv \\
    --dataset_base_path ./data_wan \\
    --output_dir ./eval_outputs/overfit_sanity \\
    --csv_row_index 0

Vanilla-I2V-style finetune (no render branch at inference)::

  PYTHONPATH=src python scripts/eval_one_video_overfit.py ... \\
    --drop_render_conditioning

Pretrained baseline (no checkpoint)::

  PYTHONPATH=src python scripts/eval_one_video_overfit.py ... --no_render_conditioner
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import List, Optional

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WM_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
for _p in (os.path.join(_WM_ROOT, "src"), _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_world_model as ewm  # noqa: E402


def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    return ewm._find_latest_checkpoint(ckpt_dir)


def _checkpoint_block_coverage(ckpt_path: str) -> dict:
    """Which DiT block indices have tensors in render_conditioner.pt (sanity for partial finetune)."""
    sd = torch.load(ckpt_path, map_location="cpu")
    blocks: set[int] = set()
    for k in sd:
        if not k.startswith("blocks."):
            continue
        parts = k.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            blocks.add(int(parts[1]))
    return {
        "n_tensors_in_file": len(sd),
        "dit_block_indices": sorted(blocks),
        "n_dit_blocks_touched": len(blocks),
    }


def _metrics_pred_vs_gt(pred_u8: np.ndarray, gt_u8: np.ndarray) -> dict:
    """pred_u8, gt_u8: (T, H, W, 3) uint8."""
    if pred_u8.shape != gt_u8.shape:
        t = min(pred_u8.shape[0], gt_u8.shape[0])
        pred_u8 = pred_u8[:t]
        gt_u8 = gt_u8[:t]
    pf = pred_u8.astype(np.float64)
    gf = gt_u8.astype(np.float64)
    diff = pf - gf
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    if mse < 1e-12:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10((255.0**2) / mse))
    # Per-frame MSE (quick read on which timesteps match)
    per_t_mse = [float(np.mean((pred_u8[i].astype(np.float64) - gt_u8[i].astype(np.float64)) ** 2)) for i in range(pred_u8.shape[0])]
    return {
        "mse_mean": mse,
        "mae_mean": mae,
        "rmse_mean": rmse,
        "psnr_db": psnr,
        "mse_frame0": per_t_mse[0] if per_t_mse else None,
        "mse_frame_last": per_t_mse[-1] if per_t_mse else None,
        "per_frame_mse": per_t_mse,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval one CSV row + GT vs pred metrics for finetune / overfit sanity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", type=str, default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/wan_render_fsdp")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--metadata_csv", type=str, default="./data_wan/metadata_one_example.csv")
    p.add_argument("--dataset_base_path", type=str, default="./data_wan")
    p.add_argument("--output_dir", type=str, default="./eval_outputs/overfit_one_video")
    p.add_argument("--csv_row_index", type=int, default=0, help="0-based row index in metadata CSV.")
    p.add_argument("--num_frames", type=int, default=33)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", choices=("bf16", "fp16", "no"), default="bf16")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--cpu_offload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no_render_conditioner", action="store_true")
    p.add_argument("--render_gate_override", type=float, default=None)
    p.add_argument(
        "--drop_render_conditioning",
        action="store_true",
        help="Inference matches train_fsdp drop_render: no render branch (vanilla I2V DiT path).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_frames % 4 != 1:
        raise SystemExit(f"--num_frames must satisfy (num_frames - 1) % 4 == 0; got {args.num_frames}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (
        torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )

    df = pd.read_csv(args.metadata_csv)
    required = {"video", "render", "prompt"}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV must have columns {required}; got {set(df.columns)}")
    idx = int(args.csv_row_index)
    if idx < 0 or idx >= len(df):
        raise SystemExit(f"--csv_row_index {idx} out of range for CSV with {len(df)} rows")

    row = df.iloc[idx].to_dict()
    base = os.path.abspath(args.dataset_base_path)

    def _abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(base, p)

    vpath = _abs(str(row["video"]))
    rpath = _abs(str(row["render"]))
    prompt_value = row.get("prompt", "")
    prompt = "" if pd.isna(prompt_value) else str(prompt_value)
    scene = os.path.splitext(os.path.basename(vpath))[0]

    ns = SimpleNamespace(
        model_path=args.model_path,
        cpu_offload=args.cpu_offload,
    )
    pipe = ewm._build_pipeline(ns, device, dtype)

    ckpt_coverage: Optional[dict] = None
    if args.no_render_conditioner:
        ckpt_meta = {"loaded": False, "reason": "--no_render_conditioner"}
    else:
        ckpt_path = args.checkpoint_path or _find_latest_checkpoint(args.checkpoint_dir)
        if ckpt_path is None:
            print(f"[overfit-eval] WARNING: no render_conditioner.pt under {args.checkpoint_dir}")
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt_coverage = _checkpoint_block_coverage(ckpt_path)
            print(f"[overfit-eval] checkpoint DiT block indices present: {ckpt_coverage['dit_block_indices']}")
        ckpt_meta = ewm._maybe_load_render_conditioner(pipe, ckpt_path, device, dtype)
    if args.render_gate_override is not None:
        with torch.no_grad():
            pipe.transformer.render_gate.data.fill_(float(args.render_gate_override))
        ckpt_meta["render_gate_overridden"] = True
        ckpt_meta["render_gate_value_after_override"] = float(pipe.transformer.render_gate.float().item())

    gt_frames = ewm._resize_pils(ewm._load_video_frames(vpath, args.num_frames), args.height, args.width)
    first_frame = gt_frames[0]
    render_frames: Optional[List] = None
    if not args.drop_render_conditioning:
        render_frames = ewm._resize_pils(ewm._load_video_frames(rpath, args.num_frames), args.height, args.width)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
        if dtype in (torch.bfloat16, torch.float16)
        else nullcontext()
    )

    call_kw = dict(
        image=first_frame,
        prompt=prompt,
        negative_prompt="",
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        generator=gen,
        output_type="np",
        return_dict=False,
    )
    if args.drop_render_conditioning:
        call_kw["drop_render_conditioning"] = True
    else:
        call_kw["render_video"] = render_frames

    t0 = time.time()
    with torch.inference_mode(), autocast_ctx:
        out = pipe(**call_kw)
    elapsed = time.time() - t0

    pred_u8 = ewm._frames_to_uint8(out[0])
    gt_u8 = ewm._frames_to_uint8(np.stack([np.asarray(f) for f in gt_frames]))
    metrics = _metrics_pred_vs_gt(pred_u8, gt_u8)

    os.makedirs(args.output_dir, exist_ok=True)
    # Optional third strip: render (only when render was used)
    strips: List[np.ndarray] = [
        ewm._label_frames(gt_u8, "GT (train target)"),
        ewm._label_frames(pred_u8, "prediction"),
    ]
    if render_frames is not None:
        rd_u8 = ewm._frames_to_uint8(np.stack([np.asarray(f) for f in render_frames]))
        strips.insert(2, ewm._label_frames(rd_u8, "render (condition)"))
    stack = ewm._hstack_rows(*strips)
    out_mp4 = os.path.join(args.output_dir, f"overfit_idx{idx:03d}_{scene}.mp4")
    writer = imageio.get_writer(
        out_mp4, fps=args.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None
    )
    try:
        for fr in stack:
            writer.append_data(fr)
    finally:
        writer.close()

    summary = {
        "scene": scene,
        "csv_row_index": idx,
        "video_path": vpath,
        "render_path": None if args.drop_render_conditioning else rpath,
        "elapsed_s": round(elapsed, 2),
        "out_mp4": out_mp4,
        "out_json": os.path.join(args.output_dir, f"overfit_idx{idx:03d}_{scene}.json"),
        "checkpoint": ckpt_meta,
        "args": vars(args),
        "metrics_vs_gt_rgb": metrics,
        "checkpoint_block_coverage": ckpt_coverage,
        "interpretation": {
            "overfit_hint": "Compare mse_mean to the same run with --no_render_conditioner (same seed). "
            "If they are nearly identical, the checkpoint may not be affecting sampling.",
            "rgb_mse_vs_gt_is_harsh": "Flow finetune minimizes one-step velocity error on LATENTS at random t, "
            "not full-chain pixel regression to GT RGB. Expect very low mse_frame0 (first frame is conditioned) "
            "and much higher mse on later frames even when training loss dropped. That pattern alone does NOT "
            "prove the finetune failed.",
            "better_overfit_signals": [
                "Training curves: train/loss_flow and train/loss_flow_ema should fall on your 2-video run.",
                "Single-step latent MSE (same as training) on a cached batch — not implemented here; "
                "compare epoch_0 vs epoch_N checkpoints that way if you add a probe.",
                "Visuals: GT | pred side-by-side for motion plausibility; LPIPS / perceptual metrics vs raw MSE.",
            ],
            "frame0_note": "mse_frame0 near zero is mostly I2V conditioning, not evidence of memorizing the clip.",
        },
    }
    with open(summary["out_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(
        json.dumps(
            {k: summary[k] for k in ("scene", "out_mp4", "out_json", "elapsed_s", "metrics_vs_gt_rgb")},
            indent=2,
        )
    )
    print(f"[overfit-eval] wrote {out_mp4}")
    print(f"[overfit-eval] wrote {summary['out_json']}")


if __name__ == "__main__":
    main()
