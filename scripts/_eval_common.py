"""
Shared utilities for eval / generation scripts.

  * ``verify_eval_matches_training``  — assert critical config fields used at
    eval match the args saved by the training run, so we don't silently load
    a checkpoint into the wrong architecture.
  * ``compute_psnr_ssim``              — per-frame PSNR + SSIM averaged over
    a (B,C,T,H,W) generated vs reference video pair.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


# Fields where any difference between training and eval IS a real problem
# (architecture / shape / dtype mismatch).
_CRITICAL_FIELDS = (
    "model_path",
    "num_frames",
    "height",
    "width",
    "mixed_precision",
    "max_sequence_length",
    "use_embodiment_adapter",
)
# These need a structural-equality check (dict, not scalar).
_DEEP_FIELDS = ("embodiment_kwargs",)
# Different at eval is fine but worth a warning so the user knows what they're
# comparing (e.g. evaluating a model trained with empty prompts on real prompts).
_WARN_FIELDS = ("ignore_prompts",)


def _resolve_train_args_for_ckpt(ckpt_path: str) -> Optional[Path]:
    """Find ``args.json`` saved by ``_setup_run_dir_and_logging`` for this ckpt.

    The training run layout is ``<output_dir>/run_<ts>/{args.json, epoch_N/render_conditioner.pt}``.
    Walk up from the ckpt to find ``args.json``.
    """
    p = Path(ckpt_path).resolve()
    for parent in (p.parent, p.parent.parent, p.parent.parent.parent):
        cand = parent / "args.json"
        if cand.is_file():
            return cand
    return None


def verify_eval_matches_training(
    eval_cfg: Dict[str, Any],
    ckpt_path: str,
    *,
    strict: bool = True,
    log: callable = print,
) -> None:
    """Compare critical fields between the eval config and the training-time
    args.json saved alongside the checkpoint. Raises on mismatch when
    ``strict=True`` (the default).

    Parameters
    ----------
    eval_cfg
        The dict loaded from the eval-time JSON (typically the SAME training
        JSON passed via ``--config``, but a user might pass a different one).
    ckpt_path
        Path to ``render_conditioner.pt`` whose run-dir contains args.json.
    strict
        If True, raises ``RuntimeError`` on any critical mismatch. WARN-only
        fields never raise.
    """
    train_args_path = _resolve_train_args_for_ckpt(ckpt_path)
    if train_args_path is None:
        log(
            f"[verify] WARN: could not find args.json near {ckpt_path}. "
            "Skipping eval/training arg consistency check."
        )
        return

    with open(train_args_path) as f:
        train_args = json.load(f)

    log(f"[verify] comparing eval config against {train_args_path}")

    mismatches: list = []
    warns: list = []

    for k in _CRITICAL_FIELDS:
        ev = eval_cfg.get(k)
        tr = train_args.get(k)
        if ev != tr:
            mismatches.append((k, tr, ev))

    for k in _DEEP_FIELDS:
        ev = eval_cfg.get(k)
        tr = train_args.get(k)
        # Compare via JSON dump to handle nested dict ordering.
        if json.dumps(ev, sort_keys=True) != json.dumps(tr, sort_keys=True):
            mismatches.append((k, tr, ev))

    for k in _WARN_FIELDS:
        ev = eval_cfg.get(k)
        tr = train_args.get(k)
        if ev != tr:
            warns.append((k, tr, ev))

    width = max((len(k) for k, _, _ in (mismatches + warns)), default=0)
    width = max(width, 14)

    if mismatches:
        log("[verify] ✗ CRITICAL mismatches between training args and eval config:")
        for k, tr, ev in mismatches:
            log(f"           {k:<{width}}  train={tr!r}  eval={ev!r}")

    if warns:
        log("[verify] △ non-critical differences (loadable but worth knowing):")
        for k, tr, ev in warns:
            log(f"           {k:<{width}}  train={tr!r}  eval={ev!r}")

    if mismatches and strict:
        raise RuntimeError(
            f"Eval config differs from training args.json on critical fields: "
            f"{[k for k, _, _ in mismatches]}. "
            f"Either pass the matching training config via --config, or run "
            f"with --no_strict_args to bypass this check."
        )

    if not mismatches and not warns:
        log("[verify] ✓ eval config matches training args on all critical fields.")


# ─────────────────────────────────────────────────────────────────────────────
# PSNR / SSIM
# ─────────────────────────────────────────────────────────────────────────────

def _to_uint8_thwc(video: torch.Tensor) -> np.ndarray:
    """(B=1, C=3, T, H, W) in [-1, 1] → (T, H, W, 3) uint8 numpy."""
    v = video[0].clamp(-1, 1).float()
    v = (v + 1.0) * 127.5
    v = v.clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0)  # (T, H, W, 3)
    return v.cpu().numpy()


def compute_psnr_ssim(
    generated: torch.Tensor,
    reference: torch.Tensor,
    *,
    skip_first_frame: bool = True,
) -> Dict[str, float]:
    """Per-frame PSNR + SSIM averaged across frames.

    Both inputs (B=1, C=3, T, H, W) tensors in [-1, 1] (the VAE postprocess
    output range). Compares pixel-by-pixel after converting to uint8 RGB.

    ``skip_first_frame=True`` (default): excludes frame 0 from the average.
    For Wan I2V, frame 0 is the **conditioning image**, NOT a model
    prediction — the inference loop explicitly preserves it
    (``latents[:, :, 0:1] = first``). Including it makes PSNR=∞ (identical
    pixels) and saturates the SSIM mean, masking the real model error on
    the predicted frames 1..T-1.

    SSIM is computed in greyscale to match common video-quality benchmarks
    (luminance is the dominant signal). PSNR is on full RGB.
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio as _psnr
        from skimage.metrics import structural_similarity as _ssim
    except ImportError as e:
        raise ImportError(
            "PSNR/SSIM require scikit-image. Install with `pip install scikit-image`."
        ) from e

    gen = _to_uint8_thwc(generated)
    ref = _to_uint8_thwc(reference)
    T_gen, T_ref = gen.shape[0], ref.shape[0]
    T = min(T_gen, T_ref)
    gen = gen[:T]
    ref = ref[:T]

    start = 1 if (skip_first_frame and T >= 2) else 0
    psnr_vals = []
    ssim_vals = []
    for t in range(start, T):
        psnr_vals.append(float(_psnr(ref[t], gen[t], data_range=255)))
        # Greyscale SSIM. Use win_size=7 (default) and explicit data_range to
        # avoid skimage warnings.
        gen_grey = gen[t].mean(axis=-1).astype(np.float32)
        ref_grey = ref[t].mean(axis=-1).astype(np.float32)
        ssim_vals.append(float(_ssim(ref_grey, gen_grey, data_range=255)))

    if not psnr_vals:
        # Pathological case (T < 2 with skip_first_frame=True). Surface NaN
        # rather than silently averaging an empty list.
        return {
            "psnr_mean": float("nan"), "psnr_min": float("nan"),
            "ssim_mean": float("nan"), "ssim_min": float("nan"),
            "n_frames":  0, "skipped_first_frame": skip_first_frame,
        }

    return {
        "psnr_mean": float(np.mean(psnr_vals)),
        "psnr_min":  float(np.min(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_min":  float(np.min(ssim_vals)),
        "n_frames":  int(T - start),
        "skipped_first_frame": skip_first_frame,
    }
