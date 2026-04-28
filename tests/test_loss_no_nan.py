"""
NaN/Inf smoke test for the wan_flow training loop.

Builds a *tiny* render-conditioned WanTransformerRenderConditioned (so it
fits on a single GPU in seconds), mimics the precompute -> training step
pipeline of train.py / train_fsdp.py, and verifies:

  1. every loss component (loss_flow, loss_tracks, loss) is finite
  2. every gradient (grad_norm) is finite
  3. parameters don't go NaN/Inf after stepping the optimizer
  4. the runtime NaN/Inf guard correctly skips a poisoned step

Run::

  PYTHONPATH=src python tests/test_loss_no_nan.py
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F


def _cfg_small():
    """Tiny DiT config that mirrors Wan's structure but is ~70x smaller."""
    return dict(
        patch_size=(1, 2, 2),
        num_attention_heads=4,
        attention_head_dim=32,
        in_channels=36,            # 16 noisy + 4 mask + 16 first-frame VAE = matches Wan I2V
        out_channels=16,
        text_dim=128,
        freq_dim=256,
        ffn_dim=512,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        added_kv_proj_dim=128,
        rope_max_seq_len=1024,
        image_dim=128,             # enables WanImageEmbedding for I2V image conditioning
        pos_embed_seq_len=None,    # matches real Wan I2V config; takes (1, 257, image_dim) directly
        # render-conditioner extras
        render_latent_channels=16,
        render_encoder_kwargs={"hidden_dim": 64},
        tracks_head_kwargs={"hidden_dim": 64, "num_freqs_xy": 4, "num_freqs_t": 4},
    )


def _make_synthetic_cache(num_samples=2, T=5, H=64, W=64, num_query_pts=8, device="cuda", dtype=torch.bfloat16):
    """Random tensors with the shapes the real cache produces."""
    T_lat = (T - 1) // 4 + 1
    Hl, Wl = H // 8, W // 8
    cache = []
    for _ in range(num_samples):
        cache.append(dict(
            clean_latents=torch.randn(1, 16, T_lat, Hl, Wl).to(dtype),
            render_latents=torch.randn(1, 16, T_lat, Hl, Wl).to(dtype),
            prompt_embeds=torch.randn(1, 16, 128).to(dtype),
            image_embeds=torch.randn(1, 257, 128).to(dtype),
            condition=torch.randn(1, 20, T_lat, Hl, Wl).to(dtype),
            gt_xy=(torch.rand(T, num_query_pts, 2) * 2 - 1).to(torch.float32),
            gt_vis=(torch.rand(T, num_query_pts) > 0.2).to(torch.float32),
        ))
    return cache, T_lat


def _build_dit(dtype, device):
    from world_model.wan_flow.model import WanTransformerRenderConditioned
    cfg = _cfg_small()
    re_kw = cfg.pop("render_encoder_kwargs")
    th_kw = cfg.pop("tracks_head_kwargs")
    dit = WanTransformerRenderConditioned(
        **cfg,
        render_encoder_kwargs=re_kw,
        tracks_head_kwargs=th_kw,
    )
    dit = dit.to(dtype=dtype, device=device)
    return dit


def _step_loss(dit, cached, scheduler, T, dtype, device, lambda_tracks=0.1, max_query_points=-1, ref_frame=0):
    """One training step's loss computation, mirroring train.py exactly."""
    clean_latents = cached["clean_latents"].to(device=device, dtype=dtype)
    render_latents = cached["render_latents"].to(device=device, dtype=dtype)
    prompt_embeds = cached["prompt_embeds"].to(device=device, dtype=dtype)
    image_embeds = cached["image_embeds"].to(device=device, dtype=dtype)
    condition = cached["condition"].to(device=device, dtype=dtype)
    gt_xy = cached["gt_xy"].to(device=device, dtype=torch.float32).unsqueeze(0)   # (1, T, N, 2)
    gt_vis = cached["gt_vis"].to(device=device, dtype=torch.float32).unsqueeze(0)  # (1, T, N)
    if max_query_points > 0 and gt_xy.shape[2] > max_query_points:
        sel = torch.randperm(gt_xy.shape[2], device=device)[:max_query_points]
        gt_xy = gt_xy[:, :, sel]
        gt_vis = gt_vis[:, :, sel]
    query_xy = gt_xy[:, ref_frame]  # (1, N, 2)

    noise = torch.randn_like(clean_latents)
    idx = torch.randint(0, len(scheduler.timesteps), (1,))
    timestep_scalar = scheduler.timesteps[idx].to(device=device).float()
    timestep_batch = timestep_scalar.expand(clean_latents.shape[0])
    noisy_latents = scheduler.scale_noise(clean_latents, timestep_batch, noise)
    noisy_latents[:, :, 0:1] = clean_latents[:, :, 0:1]

    latent_model_input = torch.cat([noisy_latents, condition], dim=1)
    target = noise - clean_latents

    out = dit(
        hidden_states=latent_model_input,
        timestep=timestep_batch,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_image=image_embeds,
        render_latents=render_latents,
        query_xy=query_xy,
        track_T=int(gt_xy.shape[1]),
        return_dict=False,
    )
    pred = out[0]
    pred_tracks = out[1] if len(out) > 1 else None

    loss_flow = F.mse_loss(pred[:, :, 1:].float(), target[:, :, 1:].float())
    if pred_tracks is not None:
        diff = F.smooth_l1_loss(pred_tracks.float(), gt_xy.float(), reduction="none").sum(-1)
        vis_sum = gt_vis.sum().clamp(min=1.0)
        loss_tracks = (diff * gt_vis).sum() / vis_sum
    else:
        loss_tracks = torch.zeros((), device=device, dtype=loss_flow.dtype)
    loss = loss_flow + lambda_tracks * loss_tracks
    return loss, loss_flow, loss_tracks


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[smoke] device={device}  dtype={dtype}")

    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    scheduler.set_timesteps(50, device=device)

    cache, T_lat = _make_synthetic_cache(num_samples=2, T=5, H=64, W=64, device=device, dtype=dtype)
    print(f"[smoke] synthetic cache: {len(cache)} samples, T_lat={T_lat}")

    dit = _build_dit(dtype=dtype, device=device)
    dit.train()
    n_params = sum(p.numel() for p in dit.parameters())
    print(f"[smoke] tiny DiT built: {n_params/1e6:.2f}M params")

    optim = torch.optim.AdamW([p for p in dit.parameters() if p.requires_grad], lr=1e-4)

    # ---- normal steps: assert all losses + grads are finite ----
    n_steps = 5
    failures = []
    for step in range(n_steps):
        cached = cache[step % len(cache)]
        loss, loss_flow, loss_tracks = _step_loss(dit, cached, scheduler, T=5, dtype=dtype, device=device)
        finite = {
            "loss": torch.isfinite(loss).item(),
            "loss_flow": torch.isfinite(loss_flow).item(),
            "loss_tracks": torch.isfinite(loss_tracks).item(),
        }
        if not all(finite.values()):
            failures.append(f"step {step}: non-finite loss components: {finite}")
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in dit.parameters() if p.grad is not None],
            max_norm=1.0,
        )
        if not torch.isfinite(gn):
            failures.append(f"step {step}: non-finite grad_norm = {float(gn):.4g}")
            optim.zero_grad(set_to_none=True)
            continue
        optim.step()
        optim.zero_grad(set_to_none=True)
        any_nan_param = any(
            (not torch.isfinite(p).all().item()) for p in dit.parameters() if p.requires_grad
        )
        if any_nan_param:
            failures.append(f"step {step}: param became non-finite after optimizer.step()")
        print(f"[smoke] step {step}: loss={float(loss):.4f}  flow={float(loss_flow):.4f}  "
              f"tracks={float(loss_tracks):.4f}  gn={float(gn):.4f}")

    # ---- adversarial step: poison the cache with NaN, ensure the guard catches it ----
    print("[smoke] testing NaN guard with poisoned cache ...")
    poisoned = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in cache[0].items()}
    poisoned["clean_latents"][:] = float("nan")
    loss_p, lf_p, lt_p = _step_loss(dit, poisoned, scheduler, T=5, dtype=dtype, device=device)
    is_finite = bool(torch.isfinite(loss_p).item())
    if is_finite:
        failures.append("poisoned step: loss was unexpectedly finite (NaN should propagate)")
    else:
        print(f"[smoke] guard correctly detected non-finite loss: {float(loss_p)}")

    if failures:
        print("[smoke] FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("[smoke] PASSED: all losses + grads + params remained finite, "
          "and NaN guard correctly fires on a poisoned step.")


if __name__ == "__main__":
    main()
