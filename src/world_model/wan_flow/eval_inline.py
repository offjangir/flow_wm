"""
In-training eval helpers — called by ``train_fsdp.py`` after each epoch.

Two granularities of eval:

  1. ``flow_mse_eval`` (every epoch, ~30 s): forward-MSE on N held-out samples
     at K random timesteps each. No VAE decode. Gives a clean per-epoch
     "is the model generalizing?" trend.

  2. ``video_eval`` (every save epoch, ~5–10 min): full denoising loop +
     VAE decode + PSNR/SSIM against ground truth, on a few held-out and a
     few training samples. Requires VAE to be re-loaded on rank 0; freed
     immediately after.

FSDP considerations:
  - During inference, ranks > 0 sit at a collective barrier inside
    ``FSDP.summon_full_params(..., rank0_only=True)``. Only rank 0 holds
    the full unsharded params and runs the forward pass.
  - The eval embeddings (text, image, render_latents, actions, etc.) are
    precomputed ONCE at training start using the same sharded
    ``_precompute_embeddings_sharded`` machinery, then everyone keeps a
    copy in CPU memory. Cheap; the held-out eval set is small.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Add the repo's ``scripts/`` to sys.path so we can import the inference
# primitives (``_denoise``, ``_vae_decode``, ``compute_psnr_ssim``) that are
# defined there. eval_inline.py lives at src/world_model/wan_flow/, so the
# scripts dir is three parents up + /scripts.
_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@torch.no_grad()
def flow_mse_eval(
    dit: torch.nn.Module,
    eval_cache: List[Dict[str, torch.Tensor]],
    scheduler,
    device: torch.device,
    dtype: torch.dtype,
    num_timesteps_per_sample: int = 4,
    seed: int = 1234,
) -> Dict[str, float]:
    """Held-out flow MSE.

    For each sample × random_t, run the DiT forward, compute MSE(pred, target).
    Returns: ``{"eval/heldout_flow_mse_mean", "eval/heldout_flow_mse_q{0..3}"}``
    where ``q*`` is the per-timestep-quartile average (same buckets as training).
    """
    n_t = len(scheduler.timesteps)
    bucket_edges = [n_t * k // 4 for k in range(5)]
    bucket_sum = [0.0, 0.0, 0.0, 0.0]
    bucket_cnt = [0, 0, 0, 0]
    all_losses: List[float] = []

    g = torch.Generator(); g.manual_seed(int(seed))

    for c in eval_cache:
        clean = c["clean_latents"].to(device=device, dtype=dtype, non_blocking=True)
        cond  = c["condition"].to(device=device, dtype=dtype, non_blocking=True)
        prompt = c["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        image  = c["image_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        rr     = c["render_latents"].to(device=device, dtype=dtype, non_blocking=True)
        actions = (
            c["actions"].to(device=device, dtype=dtype, non_blocking=True).unsqueeze(0)
            if "actions" in c else None
        )
        for _ in range(num_timesteps_per_sample):
            noise = torch.randn(clean.shape, generator=g).to(device=device, dtype=dtype)
            t_idx = int(torch.randint(0, n_t, (1,), generator=g).item())
            t_scalar = scheduler.timesteps[t_idx].to(device=device).float()
            t_batch  = t_scalar.expand(clean.shape[0])
            noisy = scheduler.scale_noise(clean, t_batch, noise)
            noisy[:, :, 0:1] = clean[:, :, 0:1]
            latent_in = torch.cat([noisy, cond], dim=1)
            target = noise - clean
            fwd_kw = dict(
                hidden_states=latent_in, timestep=t_batch,
                encoder_hidden_states=prompt, encoder_hidden_states_image=image,
                render_latents=rr, return_dict=False,
            )
            if actions is not None:
                fwd_kw["actions"] = actions
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                pred = dit(**fwd_kw)[0]
            loss = float(F.mse_loss(pred[:, :, 1:].float(), target[:, :, 1:].float()))
            all_losses.append(loss)
            b = min(3, max(0, sum(1 for e in bucket_edges[1:] if t_idx >= e)))
            bucket_sum[b] += loss
            bucket_cnt[b] += 1

    out = {
        "eval/heldout_flow_mse_mean": float(np.mean(all_losses)) if all_losses else float("nan"),
        "eval/heldout_n_samples": float(len(eval_cache)),
        "eval/heldout_n_evals":   float(len(all_losses)),
    }
    for k in range(4):
        if bucket_cnt[k] > 0:
            out[f"eval/heldout_flow_mse_q{k}"] = bucket_sum[k] / bucket_cnt[k]
    return out


def _compute_psnr_ssim(generated: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    """Per-frame mean PSNR/SSIM. Both inputs in [-1, 1], shape (B,C,T,H,W)."""
    from _eval_common import compute_psnr_ssim
    return compute_psnr_ssim(generated, reference)


@torch.no_grad()
def video_eval(
    dit: torch.nn.Module,
    vae,                            # WanVAEChunkedEncode-like (already on device)
    eval_cache: List[Dict[str, torch.Tensor]],
    scheduler,
    device: torch.device,
    dtype: torch.dtype,
    num_inference_steps: int = 30,
    cfg_scale: float = 1.0,
    seed: int = 4242,
    tag: str = "heldout",
) -> Dict[str, float]:
    """Full denoising loop + VAE decode + PSNR/SSIM against clean reference.

    Returns: ``{"eval/{tag}_psnr_mean", "eval/{tag}_ssim_mean", "eval/{tag}_n_samples"}``.
    ``tag`` distinguishes held-out vs train evals when both are run.
    """
    # Import locally to avoid bringing the heavy VAE machinery into module
    # import-time scope.
    from generate_videos import _denoise, _vae_decode

    psnrs, ssims = [], []
    scheduler.set_timesteps(num_inference_steps, device=device)
    for i, c in enumerate(eval_cache):
        clean = c["clean_latents"].to(device=device, dtype=dtype, non_blocking=True)
        rl    = c["render_latents"].to(device=device, dtype=dtype, non_blocking=True)
        # _denoise takes the cached dict directly (prompt/image/condition/actions).
        latents = _denoise(
            dit, scheduler,
            clean_latents_for_first_frame=clean,
            cached=c,
            render_latents=rl,
            num_inference_steps=num_inference_steps,
            device=device, dtype=dtype,
            seed=seed + i, cfg_scale=cfg_scale,
        )
        gen = _vae_decode(vae, latents)
        ref = _vae_decode(vae, clean)
        m = _compute_psnr_ssim(gen, ref)
        psnrs.append(float(m["psnr_mean"]))
        ssims.append(float(m["ssim_mean"]))

    return {
        f"eval/{tag}_psnr_mean":  float(np.mean(psnrs)) if psnrs else float("nan"),
        f"eval/{tag}_ssim_mean":  float(np.mean(ssims)) if ssims else float("nan"),
        f"eval/{tag}_n_samples":  float(len(eval_cache)),
    }


__all__ = ["flow_mse_eval", "video_eval"]
