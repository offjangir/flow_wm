"""
v2 collapse diagnostic — runs 4 probes on a v2 ``ActionConditionedTembAdapter``
checkpoint, dumps scalar metrics to JSON, and saves a 6-panel summary PNG.

Probes
------
1. **Action-shuffle test**: compare contribution(render_i, action_i) vs
   contribution(render_i, action_j) for j != i. If the model ignores actions,
   the delta is near zero — proves cross-attn output is functionally
   action-independent.
2. **SVD of out_proj.weight**: singular value spectrum exposes effective rank.
   A near-rank-1 spectrum + a dominant left-singular vector aligned with the
   per-scene mean of contributions ⇒ all contributions live on one line.
3. **V (action value) per-scene cos-sim**: each scene's mean V vector
   (action_encoder → cross_attn.W_v). If V is near-constant across scenes,
   the cross-attn output is forced into span(V) — the bias-direction trap.
4. **Per-scene contribution mean R-ratio + cos-sim**: classic shared/per-scene
   magnitude ratio + pairwise cos-sim of per-scene contribution means.
   R >> 1 and cos near 1 means the model emits a shared signal regardless of
   input.

Run
---
    PYTHONPATH=src python scripts/diagnose_v2_collapse.py \
        --ckpt   checkpoints/wan_render_drrobot_1k_legacy_8xl40_action_aware_v2/run_20260507_203342/epoch_29/render_conditioner.pt \
        --eval-csv data_wan_1k/train_metadata_test.csv \
        --n-scenes 16 \
        --out-dir  diagnostics/v2_epoch29

Output
------
    {out_dir}/metrics.json   — scalar metrics from each probe
    {out_dir}/diag.png       — 6-panel summary figure
    {out_dir}/raw.npz        — raw per-scene means / SVD vectors (for further analysis)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_model.wan_flow.embodiment_adapter_v2 import ActionConditionedTembAdapter
from world_model.wan_flow.data import RenderI2VMetadataDataset
from world_model.wan_flow.train import _precompute_embeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="path to render_conditioner.pt with action_adapter.* keys")
    p.add_argument("--eval-csv", required=True)
    p.add_argument("--dataset-base", default="data_wan_1k")
    p.add_argument("--action-stats", default="data_wan_1k/action_stats.json")
    p.add_argument("--action-field", default="state")
    p.add_argument("--model-path", default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    p.add_argument("--n-scenes", type=int, default=16)
    p.add_argument("--n-frames", type=int, default=81)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--width", type=int, default=432)
    p.add_argument("--action-dim", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--inner-dim", type=int, default=5120)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    return p.parse_args()


def load_v2_adapter(args, device, dtype) -> ActionConditionedTembAdapter:
    """Instantiate v2 adapter and load only action_adapter.* keys from ckpt."""
    adapter = ActionConditionedTembAdapter(
        action_dim=args.action_dim,
        inner_dim=args.inner_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        spatial_downsample=2,
    )
    sd_full = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    prefix = "action_adapter."
    sd = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
    missing, unexpected = adapter.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN missing: {missing[:6]}{'...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"  WARN unexpected: {unexpected[:6]}")
    print(f"  loaded {len(sd)} tensors from {args.ckpt}")
    return adapter.to(device=device, dtype=dtype).eval()


def build_eval_cache(args, device, dtype) -> List[dict]:
    """Use the same precompute helper as training to build N held-out samples."""
    ds = RenderI2VMetadataDataset(
        base_path=args.dataset_base,
        metadata_csv=args.eval_csv,
        num_frames=args.n_frames,
        height=args.height,
        width=args.width,
        repeat=1,
        action_stats_path=args.action_stats,
        action_field=args.action_field,
    )
    ds.rows = ds.rows[: args.n_scenes]
    print(f"[precompute] {len(ds.rows)} scenes from {args.eval_csv}")
    encoder_dtype = torch.bfloat16
    cache, _, _ = _precompute_embeddings(
        model_path=args.model_path, dataset=ds,
        device=device, height=args.height, width=args.width,
        num_frames=args.n_frames,
        max_seq_len=256, encoder_dtype=encoder_dtype,
        ignore_prompts=True, cache_path=None, safe_loading=True,
    )
    return cache


@torch.no_grad()
def adapter_forward_with_hooks(
    adapter: ActionConditionedTembAdapter,
    render_latents: torch.Tensor,
    actions: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run the v2 adapter forward and capture Q, K, V, enriched, contribution.

    Returns a dict of tensors detached on CPU in fp32.
    """
    captured: Dict[str, torch.Tensor] = {}

    # Hook q_norm OUTPUT (Q after layer norm — the actual MHA query).
    h_q  = adapter.q_norm.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("Q", out.detach()))
    # Hook kv_norm OUTPUT (action features after layer norm — input to MHA's
    # internal Wk / Wv).
    h_kv = adapter.kv_norm.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("KV_norm", out.detach()))
    # Hook out_norm OUTPUT == enriched after LN (what feeds into out_proj).
    h_e  = adapter.out_norm.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("enriched_normed", out.detach()))

    try:
        contribution = adapter(render_latents, actions)
    finally:
        h_q.remove(); h_kv.remove(); h_e.remove()

    # Extract internal K, V from cross_attn.in_proj_weight applied to KV_norm.
    # in_proj_weight is (3*D, D), stacked as [Wq, Wk, Wv].
    D = adapter.inner_dim
    W = adapter.cross_attn.in_proj_weight
    b = adapter.cross_attn.in_proj_bias
    kv = captured["KV_norm"].to(W.dtype)
    K = F.linear(kv, W[D:2*D, :], b[D:2*D] if b is not None else None)
    V = F.linear(kv, W[2*D:3*D, :], b[2*D:3*D] if b is not None else None)

    return {
        "Q": captured["Q"].float().cpu(),                       # (B, T*S, D)
        "K": K.float().cpu(),                                   # (B, T_act, D)
        "V": V.float().cpu(),                                   # (B, T_act, D)
        "enriched_normed": captured["enriched_normed"].float().cpu(),
        "contribution": contribution.float().cpu(),             # (B, T*S, D)
    }


def per_scene_stats(
    adapter: ActionConditionedTembAdapter,
    cache: List[dict],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, np.ndarray]:
    """For each scene, run adapter(render_i, action_i) AND adapter(render_i, action_j)
    where j is the scene shuffled by +1. Return per-scene means + raw shuffle deltas."""
    N = len(cache)
    contributions_real: List[torch.Tensor] = []
    V_real_means: List[torch.Tensor] = []
    contributions_shuffled: List[torch.Tensor] = []
    actions_list = [c["actions"].to(device, dtype).unsqueeze(0) for c in cache]
    print(f"[forward] running {N} real + {N} shuffled passes ...")
    for i, c in enumerate(cache):
        render = c["render_latents"].to(device=device, dtype=dtype, non_blocking=True)
        a_real = actions_list[i]
        a_shuf = actions_list[(i + 1) % N]  # use next scene's actions
        out_real = adapter_forward_with_hooks(adapter, render, a_real)
        out_shuf = adapter_forward_with_hooks(adapter, render, a_shuf)
        # Per-token mean → (D,). Squeeze batch dim (B=1).
        contributions_real.append(out_real["contribution"].squeeze(0).mean(dim=0))
        contributions_shuffled.append(out_shuf["contribution"].squeeze(0).mean(dim=0))
        # V per-scene mean over T_act → (D,).
        V_real_means.append(out_real["V"].squeeze(0).mean(dim=0))
        if (i + 1) % 4 == 0:
            print(f"  [{i+1}/{N}] done")

    return {
        "contrib_real_means":     torch.stack(contributions_real).numpy(),     # (N, D)
        "contrib_shuffled_means": torch.stack(contributions_shuffled).numpy(),  # (N, D)
        "V_real_means":           torch.stack(V_real_means).numpy(),            # (N, D)
    }


def cos_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for rows of X (N, D)."""
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X_norm @ X_norm.T


def r_ratio(per_scene_means: np.ndarray) -> Tuple[float, float, float]:
    """Classic shared/per-scene magnitude ratio.

    shared = ||mean across scenes||
    per_scene_resid = mean over scenes of ||scene_mean - shared||
    R = shared / per_scene_resid
    """
    shared = per_scene_means.mean(axis=0)
    shared_mag = float(np.linalg.norm(shared))
    resid = per_scene_means - shared[None, :]
    per_scene_resid_mag = float(np.linalg.norm(resid, axis=1).mean())
    R = shared_mag / max(per_scene_resid_mag, 1e-12)
    return R, shared_mag, per_scene_resid_mag


def svd_out_proj(adapter: ActionConditionedTembAdapter) -> Dict[str, np.ndarray]:
    """SVD of v2's external out_proj.weight (D, D)."""
    W = adapter.out_proj.weight.detach().float().cpu().numpy()
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    # Effective rank (participation ratio of squared sigmas).
    p = (S ** 2)
    p = p / max(p.sum(), 1e-12)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    return {"U_top": U[:, 0], "S": S, "Vh_top": Vh[0, :], "eff_rank": np.array([eff_rank])}


def make_figure(
    out_path: Path,
    svd: Dict[str, np.ndarray],
    contrib_cos: np.ndarray,
    V_cos: np.ndarray,
    contrib_means: np.ndarray,
    shuffle_deltas: np.ndarray,
    summary: Dict[str, float],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    # 1. SVD spectrum
    ax = axes[0, 0]
    ax.semilogy(svd["S"], "o-", markersize=3)
    ax.set_title(f"out_proj.weight SVD (eff_rank={svd['eff_rank'][0]:.2f})")
    ax.set_xlabel("singular index"); ax.set_ylabel("singular value (log)")
    ax.grid(alpha=0.3)

    # 2. contribution per-scene cos-sim heatmap
    ax = axes[0, 1]
    im = ax.imshow(contrib_cos, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(f"contribution per-scene cos-sim\n(mean off-diag={summary['contrib_cos_offdiag_mean']:.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3. V per-scene cos-sim heatmap
    ax = axes[0, 2]
    im = ax.imshow(V_cos, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(f"V (action value) per-scene cos-sim\n(mean off-diag={summary['V_cos_offdiag_mean']:.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 4. PCA scatter of per-scene contribution means
    ax = axes[1, 0]
    # Center then project on top-2 PCs of contribution means.
    c = contrib_means - contrib_means.mean(axis=0, keepdims=True)
    Uc, Sc, Vhc = np.linalg.svd(c, full_matrices=False)
    proj = Uc[:, :2] * Sc[:2]
    ax.scatter(proj[:, 0], proj[:, 1], s=40)
    for i in range(proj.shape[0]):
        ax.annotate(str(i), (proj[i, 0], proj[i, 1]), fontsize=7)
    ax.set_title(f"PCA of per-scene contribution means\n(top-2 var = {(Sc[:2]**2 / (Sc**2).sum()).sum()*100:.1f}%)")
    ax.set_xlabel(f"PC1 ({Sc[0]**2 / (Sc**2).sum()*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({Sc[1]**2 / (Sc**2).sum()*100:.1f}%)")
    ax.grid(alpha=0.3)

    # 5. Action-shuffle delta histogram
    ax = axes[1, 1]
    ax.hist(shuffle_deltas, bins=20, edgecolor="black")
    ax.axvline(summary["shuffle_delta_median"], color="red", ls="--",
               label=f"median={summary['shuffle_delta_median']:.3f}")
    ax.set_title("Action-shuffle: ||c_real - c_shuf|| / ||c_real||\n(0 = actions ignored, 1 = actions decisive)")
    ax.set_xlabel("relative delta"); ax.legend(); ax.grid(alpha=0.3)

    # 6. Summary text panel
    ax = axes[1, 2]; ax.axis("off")
    lines = [
        "── v2 collapse diagnostic ──",
        "",
        f"effective rank of out_proj.weight: {summary['out_proj_eff_rank']:.2f} / {svd['S'].shape[0]}",
        f"top-1 singular value share:        {summary['out_proj_sigma1_share']*100:.1f}%",
        "",
        f"R-ratio (shared / per-scene):      {summary['R_ratio']:.2f}",
        f"contrib per-scene cos (off-diag):  {summary['contrib_cos_offdiag_mean']:.3f}",
        f"V (action) per-scene cos:          {summary['V_cos_offdiag_mean']:.3f}",
        "",
        "action-shuffle (real vs shuffled actions):",
        f"  median delta: {summary['shuffle_delta_median']:.3f}",
        f"  mean   delta: {summary['shuffle_delta_mean']:.3f}",
        f"  90th-%ile:    {summary['shuffle_delta_p90']:.3f}",
        "",
        "Interpretation:",
        "  R >> 1, cos near 1, shuffle ≈ 0  →  collapse confirmed",
        "  R ≈ 1, cos near 0, shuffle > 0.3 →  healthy",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9,
            va="top", ha="left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[plot] {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print("=== v2 collapse diagnostic ===")
    print(f"ckpt:    {args.ckpt}")
    print(f"eval:    {args.eval_csv}  n_scenes={args.n_scenes}")
    print(f"device:  {device}  dtype={dtype}")

    adapter = load_v2_adapter(args, device, dtype)
    cache = build_eval_cache(args, device, dtype)
    if len(cache) < 2:
        raise RuntimeError(f"need at least 2 scenes; got {len(cache)}")

    svd = svd_out_proj(adapter)
    stats = per_scene_stats(adapter, cache, device, dtype)

    # Per-scene metrics
    c_cos  = cos_sim_matrix(stats["contrib_real_means"])
    V_cos  = cos_sim_matrix(stats["V_real_means"])
    N = c_cos.shape[0]
    offdiag = ~np.eye(N, dtype=bool)
    R, shared_mag, per_scene_resid_mag = r_ratio(stats["contrib_real_means"])

    # Action shuffle (relative L2 distance contribution_real vs shuffled)
    c_real = stats["contrib_real_means"]
    c_shuf = stats["contrib_shuffled_means"]
    diff = np.linalg.norm(c_real - c_shuf, axis=1)
    base = np.linalg.norm(c_real, axis=1) + 1e-12
    shuffle_deltas = diff / base

    sigma1_share = float(svd["S"][0] ** 2 / max((svd["S"] ** 2).sum(), 1e-12))

    summary: Dict[str, float] = {
        "n_scenes":                int(N),
        "out_proj_eff_rank":       float(svd["eff_rank"][0]),
        "out_proj_sigma1_share":   sigma1_share,
        "contrib_cos_offdiag_mean":float(c_cos[offdiag].mean()),
        "V_cos_offdiag_mean":      float(V_cos[offdiag].mean()),
        "R_ratio":                 float(R),
        "shared_mag":              float(shared_mag),
        "per_scene_resid_mag":     float(per_scene_resid_mag),
        "shuffle_delta_mean":      float(shuffle_deltas.mean()),
        "shuffle_delta_median":    float(np.median(shuffle_deltas)),
        "shuffle_delta_p90":       float(np.quantile(shuffle_deltas, 0.9)),
        "ckpt":                    str(args.ckpt),
    }

    # Dump
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        out_dir / "raw.npz",
        contrib_real_means=stats["contrib_real_means"],
        contrib_shuffled_means=stats["contrib_shuffled_means"],
        V_real_means=stats["V_real_means"],
        out_proj_S=svd["S"],
        out_proj_U_top=svd["U_top"],
        out_proj_Vh_top=svd["Vh_top"],
        contrib_cos=c_cos,
        V_cos=V_cos,
        shuffle_deltas=shuffle_deltas,
    )
    make_figure(
        out_dir / "diag.png",
        svd, c_cos, V_cos,
        stats["contrib_real_means"],
        shuffle_deltas,
        summary,
    )

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:30s} {v}")
    print(f"\nFull report: {out_dir}/{{metrics.json, diag.png, raw.npz}}")


if __name__ == "__main__":
    main()
