"""
v4 collapse diagnostic — runs 3 probes on an
``ActionRenderSelfCrossAdapterV4`` checkpoint, dumps scalar metrics to
JSON, and saves a 6-panel summary PNG (parallel layout to
diagnose_v2_collapse.py / diagnose_egowm_collapse.py for side-by-side
comparison).

Probes
------
1. **Render-shuffle test**: compare contribution(render_i, action_i) vs
   contribution(render_j, action_i) for j != i. Measures whether the
   adapter uses the render input.
2. **Action-shuffle test**: compare contribution(render_i, action_i) vs
   contribution(render_i, action_j). Measures whether the adapter uses
   the action input.
3. **SVD of to_temb's final Linear (5120 x 2048)**: rank/spectrum check.
   Low effective rank ⇒ output lives on a low-dim subspace.
4. **Per-scene contribution mean R-ratio + cos-sim**: shared/per-scene
   magnitude ratio + pairwise cos-sim of per-scene contribution means.
   R >> 1 and cos near 1 means the adapter emits a shared signal
   regardless of input.

Run
---
    PYTHONPATH=src python scripts/diagnose_v4_collapse.py \
        --ckpt   checkpoints/wan_render_drrobot_1k_v4_full_17f_*/run_*/epoch_9/render_conditioner.pt \
        --eval-csv data_wan_1k/train_metadata_test.csv \
        --cache-path data_wan_1k/precompute_cache_probe_v4_16scenes_17f.pt \
        --n-scenes 16 \
        --out-dir diagnostics/v4_epoch9

Output
------
    {out_dir}/metrics.json   — scalar metrics from each probe
    {out_dir}/diag.png       — 6-panel summary figure
    {out_dir}/raw.npz        — raw per-scene means / SVD vectors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_model.wan_flow.embodiment_adapter_v3 import ActionRenderSelfCrossAdapterV4
from world_model.wan_flow.data import RenderI2VMetadataDataset
from world_model.wan_flow.train import _precompute_embeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="path to render_conditioner.pt with action_adapter_v4.* keys")
    p.add_argument("--eval-csv", required=True)
    p.add_argument("--dataset-base", default="data_wan_1k")
    p.add_argument("--action-stats", default="data_wan_1k/action_stats.json")
    p.add_argument("--action-field", default="state")
    p.add_argument("--model-path", default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    p.add_argument("--n-scenes", type=int, default=16)
    p.add_argument("--n-frames", type=int, default=17)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--width", type=int, default=432)
    # v4 adapter kwargs (must match the trained config)
    p.add_argument("--action-dim", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--inner-dim", type=int, default=5120)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-self-blocks", type=int, default=2)
    p.add_argument("--num-cross-blocks", type=int, default=2)
    p.add_argument("--spatial-downsample", type=int, default=2)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--max-action-frames", type=int, default=128)
    p.add_argument("--max-t", type=int, default=32)
    p.add_argument("--max-h", type=int, default=32)
    p.add_argument("--max-w", type=int, default=64)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--cache-path", default=None,
                   help="Reuse a precompute cache instead of re-running encoders. "
                        "USE A PROBE-SUFFIXED PATH — see memory:probe-cache-clobber.")
    return p.parse_args()


def load_v4_adapter(args, device, dtype) -> ActionRenderSelfCrossAdapterV4:
    adapter = ActionRenderSelfCrossAdapterV4(
        action_dim=args.action_dim,
        inner_dim=args.inner_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_self_blocks=args.num_self_blocks,
        num_cross_blocks=args.num_cross_blocks,
        spatial_downsample=args.spatial_downsample,
        ffn_mult=args.ffn_mult,
        max_action_frames=args.max_action_frames,
        max_t=args.max_t,
        max_h=args.max_h,
        max_w=args.max_w,
    )
    sd_full = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    prefix = "action_adapter_v4."
    sd = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
    missing, unexpected = adapter.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN missing: {missing[:6]}{'...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"  WARN unexpected: {unexpected[:6]}")
    print(f"  loaded {len(sd)} action_adapter_v4 tensors from {args.ckpt}")
    return adapter.to(device=device, dtype=dtype).eval()


def build_eval_cache(args, device, dtype) -> List[dict]:
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
        ignore_prompts=True, cache_path=args.cache_path, safe_loading=True,
    )
    return cache


@torch.no_grad()
def per_scene_stats(
    adapter: ActionRenderSelfCrossAdapterV4,
    cache: List[dict],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, np.ndarray]:
    """For each scene i, run adapter on:
      - (render_i, action_i)               — "real"
      - (render_{i+1 mod N}, action_i)     — "render-shuffled"
      - (render_i, action_{i+1 mod N})     — "action-shuffled"
    Return per-scene token-mean contributions (D,) on CPU fp32.
    """
    N = len(cache)
    renders = [c["render_latents"].to(device, dtype) for c in cache]
    actions = [c["actions"].to(device, dtype).unsqueeze(0) for c in cache]
    if renders[0].ndim == 5 and renders[0].shape[0] != 1:
        # ensure batch dim
        renders = [r.unsqueeze(0) if r.ndim == 4 else r for r in renders]

    contribs_real, contribs_rshuf, contribs_ashuf = [], [], []
    print(f"[forward] {N} real + {N} render-shuffled + {N} action-shuffled passes ...")
    for i in range(N):
        r_i, a_i = renders[i], actions[i]
        r_j = renders[(i + 1) % N]
        a_j = actions[(i + 1) % N]
        out_real = adapter(r_i, a_i).float().squeeze(0)
        out_rshuf = adapter(r_j, a_i).float().squeeze(0)
        out_ashuf = adapter(r_i, a_j).float().squeeze(0)
        contribs_real.append(out_real.mean(dim=0).cpu())
        contribs_rshuf.append(out_rshuf.mean(dim=0).cpu())
        contribs_ashuf.append(out_ashuf.mean(dim=0).cpu())
        if (i + 1) % 4 == 0:
            print(f"  [{i+1}/{N}] done")

    return {
        "contrib_real_means":  torch.stack(contribs_real).numpy(),
        "contrib_rshuf_means": torch.stack(contribs_rshuf).numpy(),
        "contrib_ashuf_means": torch.stack(contribs_ashuf).numpy(),
    }


def cos_sim_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def r_ratio(per_scene_means: np.ndarray) -> Tuple[float, float, float]:
    shared = per_scene_means.mean(axis=0)
    shared_mag = float(np.linalg.norm(shared))
    resid = per_scene_means - shared[None, :]
    per_scene_resid_mag = float(np.linalg.norm(resid, axis=1).mean())
    R = shared_mag / max(per_scene_resid_mag, 1e-12)
    return R, shared_mag, per_scene_resid_mag


def svd_to_temb_final(adapter: ActionRenderSelfCrossAdapterV4) -> Dict[str, np.ndarray]:
    """SVD of v4's final to_temb Linear weight (inner_dim, hidden_dim*ffn_mult)."""
    W = adapter.to_temb[-1].weight.detach().float().cpu().numpy()
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    p = S ** 2
    p = p / max(p.sum(), 1e-12)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    return {"U_top": U[:, 0], "S": S, "Vh_top": Vh[0, :], "eff_rank": np.array([eff_rank])}


def make_figure(
    out_path: Path,
    svd: Dict[str, np.ndarray],
    contrib_cos: np.ndarray,
    contrib_means: np.ndarray,
    rshuf_deltas: np.ndarray,
    ashuf_deltas: np.ndarray,
    summary: Dict[str, float],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    ax.semilogy(svd["S"], "o-", markersize=3)
    ax.set_title(f"to_temb final Linear SVD\n(eff_rank={svd['eff_rank'][0]:.2f} / {svd['S'].shape[0]})")
    ax.set_xlabel("singular index"); ax.set_ylabel("σ (log)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    im = ax.imshow(contrib_cos, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(f"contribution per-scene cos-sim\n(mean off-diag={summary['contrib_cos_offdiag_mean']:.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 2]
    mags = np.linalg.norm(contrib_means, axis=1)
    ax.bar(range(len(mags)), mags)
    ax.axhline(mags.mean(), color="red", ls="--", label=f"mean={mags.mean():.2f}")
    ax.set_title("per-scene contribution magnitude")
    ax.set_xlabel("scene"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    c = contrib_means - contrib_means.mean(axis=0, keepdims=True)
    Uc, Sc, _ = np.linalg.svd(c, full_matrices=False)
    proj = Uc[:, :2] * Sc[:2]
    ax.scatter(proj[:, 0], proj[:, 1], s=40)
    for i in range(proj.shape[0]):
        ax.annotate(str(i), (proj[i, 0], proj[i, 1]), fontsize=7)
    var2 = (Sc[:2] ** 2 / (Sc ** 2).sum()).sum() * 100
    ax.set_title(f"PCA of per-scene contribution means\n(top-2 var = {var2:.1f}%)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(rshuf_deltas, bins=15, alpha=0.6, edgecolor="black",
            label=f"render-shuf med={summary['rshuf_delta_median']:.3f}")
    ax.hist(ashuf_deltas, bins=15, alpha=0.6, edgecolor="black",
            label=f"action-shuf med={summary['ashuf_delta_median']:.3f}")
    ax.set_title("shuffle deltas (||real − shuf|| / ||real||)\n(0 = input ignored, 1 = decisive)")
    ax.set_xlabel("relative delta"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 2]; ax.axis("off")
    lines = [
        "── v4 collapse diagnostic ──",
        "",
        f"to_temb final Linear eff_rank: {summary['to_temb_eff_rank']:.2f} / {svd['S'].shape[0]}",
        f"to_temb σ1 share:              {summary['to_temb_sigma1_share']*100:.1f}%",
        "",
        f"R-ratio (shared/per-scene):    {summary['R_ratio']:.2f}",
        f"contrib per-scene cos:         {summary['contrib_cos_offdiag_mean']:.3f}",
        "",
        "render-shuffle (replace render with other scene's):",
        f"  median delta: {summary['rshuf_delta_median']:.3f}",
        f"  mean   delta: {summary['rshuf_delta_mean']:.3f}",
        f"  90th-%ile:    {summary['rshuf_delta_p90']:.3f}",
        "",
        "action-shuffle (replace action with other scene's):",
        f"  median delta: {summary['ashuf_delta_median']:.3f}",
        f"  mean   delta: {summary['ashuf_delta_mean']:.3f}",
        f"  90th-%ile:    {summary['ashuf_delta_p90']:.3f}",
        "",
        "Interpretation:",
        "  R >> 1, cos near 1, shufs ≈ 0  → collapse",
        "  R ≈ 1, cos < 0.5, shufs > 0.3  → healthy",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9, va="top", ha="left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[plot] {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print("=== v4 collapse diagnostic ===")
    print(f"ckpt:   {args.ckpt}")
    print(f"eval:   {args.eval_csv}  n_scenes={args.n_scenes}")
    print(f"device: {device}  dtype={dtype}")

    adapter = load_v4_adapter(args, device, dtype)
    cache = build_eval_cache(args, device, dtype)
    if len(cache) < 2:
        raise RuntimeError(f"need at least 2 scenes; got {len(cache)}")

    svd = svd_to_temb_final(adapter)
    stats = per_scene_stats(adapter, cache, device, dtype)

    c_cos = cos_sim_matrix(stats["contrib_real_means"])
    N = c_cos.shape[0]
    offdiag = ~np.eye(N, dtype=bool)
    R, shared_mag, per_scene_resid_mag = r_ratio(stats["contrib_real_means"])

    c_real = stats["contrib_real_means"]
    rshuf_diff = np.linalg.norm(c_real - stats["contrib_rshuf_means"], axis=1)
    ashuf_diff = np.linalg.norm(c_real - stats["contrib_ashuf_means"], axis=1)
    base = np.linalg.norm(c_real, axis=1) + 1e-12
    rshuf_deltas = rshuf_diff / base
    ashuf_deltas = ashuf_diff / base

    sigma1_share = float(svd["S"][0] ** 2 / max((svd["S"] ** 2).sum(), 1e-12))

    summary: Dict[str, float] = {
        "n_scenes":               int(N),
        "to_temb_eff_rank":       float(svd["eff_rank"][0]),
        "to_temb_sigma1_share":   sigma1_share,
        "contrib_cos_offdiag_mean":float(c_cos[offdiag].mean()),
        "R_ratio":                float(R),
        "shared_mag":             float(shared_mag),
        "per_scene_resid_mag":    float(per_scene_resid_mag),
        "rshuf_delta_mean":       float(rshuf_deltas.mean()),
        "rshuf_delta_median":     float(np.median(rshuf_deltas)),
        "rshuf_delta_p90":        float(np.quantile(rshuf_deltas, 0.9)),
        "ashuf_delta_mean":       float(ashuf_deltas.mean()),
        "ashuf_delta_median":     float(np.median(ashuf_deltas)),
        "ashuf_delta_p90":        float(np.quantile(ashuf_deltas, 0.9)),
        "ckpt":                   str(args.ckpt),
    }

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        out_dir / "raw.npz",
        contrib_real_means=stats["contrib_real_means"],
        contrib_rshuf_means=stats["contrib_rshuf_means"],
        contrib_ashuf_means=stats["contrib_ashuf_means"],
        to_temb_S=svd["S"],
        to_temb_U_top=svd["U_top"],
        to_temb_Vh_top=svd["Vh_top"],
        contrib_cos=c_cos,
        rshuf_deltas=rshuf_deltas,
        ashuf_deltas=ashuf_deltas,
    )
    make_figure(out_dir / "diag.png", svd, c_cos, c_real, rshuf_deltas, ashuf_deltas, summary)

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:30s} {v}")
    print(f"\nFull report: {out_dir}/{{metrics.json, diag.png, raw.npz}}")


if __name__ == "__main__":
    main()
