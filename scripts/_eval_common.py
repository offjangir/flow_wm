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
    # Legacy render-conditioner variant. egowm/v1/v2 instantiate different
    # submodules under different attribute names — mismatch ⇒ strict-load
    # silently drops every v2 action_adapter.* key into "unexpected".
    "legacy_render_variant",
    # v2 adapter's action_field controls which 8-d stream is fed in
    # (`state` = joint+gripper). A mismatch evaluates the model on a
    # distribution it never saw.
    "action_field",
)
# These need a structural-equality check (dict, not scalar).
_DEEP_FIELDS = (
    "embodiment_kwargs",
    # v2: action_dim, hidden_dim, num_heads, spatial_downsample. Any change
    # alters parameter shapes ⇒ load fails or loads wrong-shape weights.
    "v2_adapter_kwargs",
)
# Different at eval is fine but worth a warning so the user knows what they're
# comparing (e.g. evaluating a model trained with empty prompts on real prompts).
_WARN_FIELDS = ("ignore_prompts",)


def _variant_needs_actions(cfg: Dict[str, Any]) -> bool:
    """Whether the model variant in ``cfg`` consumes the ``actions`` stream
    at forward time. Mirrors the branching in
    ``WanTransformerRenderConditioned.forward``:

    - ``use_embodiment_adapter=True``: actions feed the embodiment when EITHER
      ``use_action_aware_adaln=True`` OR ``action_dim`` is set in
      ``embodiment_kwargs``. Otherwise the embodiment ignores them.
    - legacy variants: only ``v2`` (``ActionConditionedTembAdapter``)
      requires actions. ``egowm`` and ``v1`` ignore them.
    """
    if cfg.get("use_embodiment_adapter", False):
        ek = cfg.get("embodiment_kwargs") or {}
        return bool(ek.get("use_action_aware_adaln")) or ek.get("action_dim") is not None
    return cfg.get("legacy_render_variant", "egowm") == "v2"


def _expected_action_dim(cfg: Dict[str, Any]) -> Optional[int]:
    """Action_dim the live model will read on the forward pass, or None if
    the variant doesn't consume actions."""
    if not _variant_needs_actions(cfg):
        return None
    if cfg.get("use_embodiment_adapter", False):
        return (cfg.get("embodiment_kwargs") or {}).get("action_dim")
    return (cfg.get("v2_adapter_kwargs") or {}).get("action_dim")


def assert_eval_data_matches_model(
    cfg: Dict[str, Any],
    dataset: Any,
    embed_cache: list,
    *,
    log: callable = print,
) -> None:
    """Audit that the eval-time data stream matches what the model variant
    will actually consume on forward.

    Fails on:
      - variant requires actions but cache lacks them (silent ablation);
      - variant requires actions and ``action_stats_path`` not set
        (cache holds raw, un-normalized actions ⇒ distribution shift);
      - cached ``actions`` action_dim != model's expected action_dim
        (shape mismatch ⇒ runtime crash or wrong weights consumed).

    Warns (no raise) on:
      - variant does NOT consume actions but cache has them (wasted I/O,
        no correctness issue — model branch ignores them).
    """
    needs = _variant_needs_actions(cfg)
    has_actions_in_cache = bool(embed_cache) and "actions" in embed_cache[0]
    variant_name = (
        "embodiment_adapter"
        if cfg.get("use_embodiment_adapter", False)
        else f"legacy:{cfg.get('legacy_render_variant', 'egowm')}"
    )
    log(f"[data-audit] variant={variant_name}  needs_actions={needs}  "
        f"cache_has_actions={has_actions_in_cache}")

    if needs and not has_actions_in_cache:
        raise RuntimeError(
            f"[data-audit] model variant {variant_name!r} consumes the actions "
            f"stream on forward, but the eval cache has none. Causes: "
            f"(a) eval CSV's 'actions' column is empty/missing; "
            f"(b) cfg.action_stats_path missing so the dataset bypassed "
            f"actions; (c) the v2 / action-aware-adaln cfg is wrong. "
            f"Forward would either raise (v2) or silently run with actions=None "
            f"(embodiment+action_aware) — neither is a valid eval."
        )

    if needs and not cfg.get("action_stats_path"):
        raise RuntimeError(
            f"[data-audit] {variant_name!r} needs actions but cfg.action_stats_path "
            f"is empty. Cache holds un-normalized actions; the model was trained "
            f"on z-scored actions, so this evaluates on a distribution shift. "
            f"Set action_stats_path to the same stats file used at training."
        )

    if needs and has_actions_in_cache:
        expected = _expected_action_dim(cfg)
        observed = int(embed_cache[0]["actions"].shape[-1])
        if expected is not None and observed != expected:
            raise RuntimeError(
                f"[data-audit] action_dim mismatch: model expects {expected} "
                f"(from cfg), cache has {observed} (from {cfg.get('action_field')!r} "
                f"field of the npz). Pick the matching action_field or rebuild "
                f"the dataset with the right one."
            )
        log(f"[data-audit] ✓ actions present in {len(embed_cache)} cache entries, "
            f"action_dim={observed}, normalized via {cfg['action_stats_path']}")

    if not needs and has_actions_in_cache:
        log(f"[data-audit] △ cache has actions but variant {variant_name!r} "
            f"ignores them on forward (harmless; only wasted precompute work).")

    if not needs and not has_actions_in_cache:
        log(f"[data-audit] ✓ variant {variant_name!r} ignores actions; cache "
            f"correctly omits them.")


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
