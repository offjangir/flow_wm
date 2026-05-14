#!/usr/bin/env python3
"""
Bias-direction diagnostic for the v2 action_adapter.

Hypothesis: out_proj learned a single direction shared across all scenes
(the "DROID domain shift"), and that's what causes the universal tint.

Test:
    For each of N held-out scenes:
        contribution_s = action_adapter(render_latents_s, actions_s)   # (1, T_tok, D)
        scene_mean_s   = contribution_s.mean(dim=(0, 1))                # (D,)
    Stack into M (S, D). Decompose into:
        shared_dir   = M.mean(dim=0)   # (D,)  — the global bias direction
        per_scene_std = M.std(dim=0)   # (D,)  — cross-scene variation per dim

    The ratio R = ||shared_dir|| / ||per_scene_std||:
        R >> 1  →  bias-direction theory CONFIRMED (one shared offset across scenes)
        R ≈ 1   →  mixed: both bias and per-scene signal exist
        R << 1  →  per-scene variation dominates  →  theory FALSIFIED

    Also report:
        - cosine-similarity matrix between scene_mean vectors (1.0 = identical direction)
        - explained-variance ratio of top PCA components on M

Usage:
    PYTHONPATH=src python scripts/probe_action_adapter_bias.py \\
        --ckpt checkpoints/wan_render_drrobot_1k_legacy_8xl40_action_aware_v2/run_20260507_203342/epoch_29/render_conditioner.pt \\
        --config configs/train_drrobot_1k_legacy_8xl40_action_aware_v2.json \\
        --eval_csv data_wan_1k/train_metadata_test.csv \\
        --num_samples 8 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SIDEGIG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIDEGIG / "src"))

from world_model.wan_flow.data import (        # noqa: E402
    RenderI2VMetadataDataset,
    _load_actions_npz,
    _load_action_stats,
)
from world_model.wan_flow.model import WanTransformerRenderConditioned  # noqa: E402
from world_model.wan_flow.train import (        # noqa: E402
    _materialize_meta_submodules,
    _precompute_embeddings,
)


def load_subset_into_model(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    target = next(model.parameters()).dtype
    sd_cast = {k: (v.to(target) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
               for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd_cast, strict=False)
    print(f"[probe] loaded {len(sd_cast) - len(unexpected)}/{len(sd_cast)} tensors  "
          f"unexpected={len(unexpected)}  missing={len(missing)} (mostly base Wan weights)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt",   required=True)
    p.add_argument("--eval_csv", required=True)
    p.add_argument("--dataset_base_path", default=None)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_npz", default="generations/v2_81f_epoch_29_bias_probe.npz")
    args = p.parse_args()

    with open(args.config) as f: cfg = json.load(f)
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[
        cfg.get("mixed_precision", "bf16")
    ]
    base = args.dataset_base_path or cfg["dataset_base_path"]

    # ── 1. Build model + load v2 ckpt ──────────────────────────────────────
    print(f"[probe] building DiT (legacy_render_variant=v2)...")
    dit = WanTransformerRenderConditioned.from_pretrained(
        cfg["model_path"], subfolder="transformer",
        torch_dtype=dtype,
        render_encoder_kwargs={},
        use_embodiment_adapter=False,
        legacy_render_variant=cfg.get("legacy_render_variant", "egowm"),
        v2_adapter_kwargs=cfg.get("v2_adapter_kwargs"),
    )
    _materialize_meta_submodules(dit)
    if hasattr(dit, "reset_zero_gates"):
        dit.reset_zero_gates()
    load_subset_into_model(dit, args.ckpt)
    dit = dit.to(device=device, dtype=dtype).eval()

    # Verify: out_proj weight non-zero (trained)
    aa = dit.action_adapter
    print(f"[probe] out_proj.weight.norm = {aa.out_proj.weight.detach().float().norm().item():.4f}  "
          f"(expect ~0.85 for trained ckpt)")

    # ── 2. Build dataset over the eval CSV ─────────────────────────────────
    eval_ds = RenderI2VMetadataDataset(
        base_path=base,
        metadata_csv=args.eval_csv,
        num_frames=cfg["num_frames"], height=cfg["height"], width=cfg["width"],
        repeat=1,
        action_stats_path=cfg.get("action_stats_path"),
        action_field=cfg.get("action_field", "state"),
    )
    n = min(args.num_samples, len(eval_ds.rows))
    eval_ds.rows = eval_ds.rows[:n]
    print(f"[probe] precomputing render_latents + actions for {n} scenes ...")

    # Run a minimal precompute to get render_latents (VAE-encoded) per scene.
    embed_cache, _z, _sched = _precompute_embeddings(
        model_path=cfg["model_path"], dataset=eval_ds,
        device=device, height=cfg["height"], width=cfg["width"],
        num_frames=cfg["num_frames"],
        max_seq_len=cfg.get("max_sequence_length", 256),
        encoder_dtype=dtype,
        ignore_prompts=bool(cfg.get("ignore_prompts", False)),
        cache_path=None, safe_loading=True,
    )

    # ── 3. Forward action_adapter ONLY on each scene ────────────────────────
    print(f"[probe] running action_adapter forward on {n} scenes ...")
    per_scene_means = []           # list of (D,)
    per_scene_norms = []           # list of scalar
    per_scene_full = []            # list of (T_tok, D) for spectral analysis

    with torch.no_grad():
        for i, c in enumerate(embed_cache):
            rl = c["render_latents"].to(device=device, dtype=dtype)
            ac = c["actions"].to(device=device, dtype=dtype).unsqueeze(0)
            contribution = aa(rl, ac)                                # (1, T*S, D)
            cf = contribution[0].float().cpu()                       # (T*S, D)
            per_scene_full.append(cf)
            per_scene_means.append(cf.mean(dim=0))                   # (D,)
            per_scene_norms.append(cf.norm(dim=-1).mean().item())    # mean per-token L2
            scene_id = eval_ds.rows[i]["video"].split("/scene_")[1].split("/")[0] \
                       if "/scene_" in eval_ds.rows[i]["video"] else f"row{i}"
            print(f"   [{i+1}/{n}] scene={scene_id}  contribution shape={tuple(contribution.shape)}  "
                  f"per-token L2 mean={per_scene_norms[-1]:.4f}")

    # ── 4. Cross-scene statistics ───────────────────────────────────────────
    M = torch.stack(per_scene_means, dim=0)           # (S, D)
    shared_dir   = M.mean(dim=0)                       # (D,)
    per_scene_std = M.std(dim=0)                       # (D,)

    norm_shared = shared_dir.norm().item()
    norm_pscene = per_scene_std.norm().item()          # Frobenius-like
    R = norm_shared / max(norm_pscene, 1e-9)

    # Cosine similarities between scene_mean vectors
    Mn = M / (M.norm(dim=-1, keepdim=True) + 1e-9)
    cos = (Mn @ Mn.T).numpy()
    upper = cos[np.triu_indices_from(cos, k=1)]

    # Centered SVD to see how concentrated variation is in top components
    centered = (M - shared_dir).float()
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    var_explained = (S.pow(2) / S.pow(2).sum()).cpu().numpy()

    # ── 5. Report ───────────────────────────────────────────────────────────
    print()
    print("================================================================")
    print(" v2 action_adapter contribution direction diagnostic")
    print("================================================================")
    print(f"  per-scene token-mean shape: {tuple(M.shape)}")
    print(f"")
    print(f"  ||shared_dir||         = {norm_shared:.4f}    (the cross-scene MEAN direction)")
    print(f"  ||per_scene_std||      = {norm_pscene:.4f}    (Frobenius of per-dim cross-scene std)")
    print(f"  ratio R = shared / std = {R:.3f}")
    print(f"")
    print(f"  cos-sim across scene-mean vectors (off-diagonal pairs):")
    print(f"    mean: {upper.mean():.4f}   median: {np.median(upper):.4f}")
    print(f"    min : {upper.min():.4f}   max:    {upper.max():.4f}")
    print(f"")
    print(f"  PCA variance explained by top components (after subtracting shared mean):")
    for i, v in enumerate(var_explained[:5]):
        print(f"    component {i+1}: {100*v:.2f}%")
    print(f"    sum top-3 : {100*var_explained[:3].sum():.2f}%")
    print(f"")
    print(f"  per-scene mean token-L2 = {np.mean(per_scene_norms):.4f}  "
          f"std={np.std(per_scene_norms):.4f}")
    print()

    # ── 6. Verdict ─────────────────────────────────────────────────────────
    print("verdict:")
    if R > 2.0:
        print(f"  R={R:.2f} >> 1   →  BIAS-DIRECTION THEORY CONFIRMED.")
        print(f"  Most of the contribution is a single direction shared across scenes.")
        print(f"  Cosine similarity {upper.mean():.3f} between scene-means supports this.")
    elif R > 0.7:
        print(f"  R={R:.2f} near 1  →  MIXED.")
        print(f"  Shared bias and per-scene variation are comparable.")
    else:
        print(f"  R={R:.2f} << 1   →  BIAS-DIRECTION THEORY FALSIFIED.")
        print(f"  Per-scene variation dominates. Contribution genuinely varies per scene.")
        print(f"  The visual tint is caused by something else.")

    # Save raw arrays for any follow-up.
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        per_scene_means=M.numpy(),
        shared_dir=shared_dir.numpy(),
        per_scene_std=per_scene_std.numpy(),
        cos_sim=cos,
        var_explained=var_explained,
        per_scene_token_l2=np.array(per_scene_norms),
        ratio_R=np.array([R]),
    )
    print(f"  raw arrays → {out_path}")


if __name__ == "__main__":
    main()
