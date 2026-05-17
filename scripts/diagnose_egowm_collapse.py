"""
egowm collapse diagnostic — runs 3 probes on an egowm
``RenderSpatialEncoder`` checkpoint, dumps scalar metrics to JSON, and saves
a 6-panel summary PNG (parallel layout to diagnose_v2_collapse.py for easy
side-by-side comparison).

Probes
------
1. **Render-shuffle test**: compare contribution(render_i) vs
   contribution(render_j) for j != i (egowm has no action input — render is
   the only conditioning, so we shuffle that). If the model ignores its
   render input, the delta is near zero — render-encoder is functionally
   input-independent.
2. **SVD of render_encoder.out.weight** (out_dim, hidden_dim): singular
   value spectrum exposes effective rank. A near-rank-1 spectrum + a
   dominant top singular vector aligned with the per-scene contribution
   mean ⇒ all contributions live on one line.
3. **Per-scene contribution mean R-ratio + cos-sim**: shared/per-scene
   magnitude ratio + pairwise cos-sim of per-scene contribution means.
   R >> 1 and cos near 1 means the encoder emits a shared signal regardless
   of input render.

(v2's probe-3 "V (action-value) per-scene cos-sim" is omitted — egowm has
no cross-attention, so there is no V to probe. The panel is kept in the
figure but marked "n/a" for layout parity.)

Run
---
    PYTHONPATH=src python scripts/diagnose_egowm_collapse.py \
        --ckpt   checkpoints/wan_render_drrobot_1k_legacy_8xl40/run_.../epoch_N/render_conditioner.pt \
        --eval-csv data_wan_1k/train_metadata_test.csv \
        --cache-path data_wan_1k/precompute_cache_17f.pt \
        --n-scenes 16 \
        --out-dir  diagnostics/egowm_epochN

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

from world_model.wan_flow.embodiment_adapter import RenderSpatialEncoder
from world_model.wan_flow.data import RenderI2VMetadataDataset
from world_model.wan_flow.train import _precompute_embeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="path to render_conditioner.pt with render_encoder.* keys")
    p.add_argument("--eval-csv", required=True)
    p.add_argument("--dataset-base", default="data_wan_1k")
    p.add_argument("--action-stats", default="data_wan_1k/action_stats.json")
    p.add_argument("--action-field", default="state")
    p.add_argument("--model-path", default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    p.add_argument("--n-scenes", type=int, default=16)
    p.add_argument("--n-frames", type=int, default=17)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--width", type=int, default=432)
    p.add_argument("--in-channels", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--out-dim", type=int, default=5120)
    p.add_argument("--spatial-pool", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--cache-path", default=None,
                   help="Reuse a precompute cache instead of re-running the "
                        "encoders. For egowm/legacy 17f use "
                        "data_wan_1k/precompute_cache_17f.pt.")
    return p.parse_args()


def load_egowm_encoder(args, device, dtype) -> RenderSpatialEncoder:
    """Instantiate egowm RenderSpatialEncoder and load only render_encoder.*
    keys from ckpt."""
    enc = RenderSpatialEncoder(
        in_channels=args.in_channels,
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        spatial_pool=args.spatial_pool,
    )
    sd_full = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    prefix = "render_encoder."
    sd = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN missing: {missing[:6]}{'...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"  WARN unexpected: {unexpected[:6]}")
    print(f"  loaded {len(sd)} render_encoder tensors from {args.ckpt}")
    return enc.to(device=device, dtype=dtype).eval()


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
    enc: RenderSpatialEncoder,
    cache: List[dict],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, np.ndarray]:
    """For each scene i, forward enc(render_i) AND enc(render_{i+1 mod N}).
    Returns per-scene contribution means (token-mean, fp32, on CPU)."""
    N = len(cache)
    contribs_real: List[torch.Tensor] = []
    contribs_shuf: List[torch.Tensor] = []
    print(f"[forward] running {N} real + {N} render-shuffled passes ...")
    renders = [c["render_latents"] for c in cache]
    for i in range(N):
        r_real = renders[i].to(device=device, dtype=dtype, non_blocking=True)
        r_shuf = renders[(i + 1) % N].to(device=device, dtype=dtype, non_blocking=True)
        out_real = enc(r_real).float().squeeze(0)   # (T*h*w, D)
        out_shuf = enc(r_shuf).float().squeeze(0)
        contribs_real.append(out_real.mean(dim=0).cpu())
        contribs_shuf.append(out_shuf.mean(dim=0).cpu())
        if (i + 1) % 4 == 0:
            print(f"  [{i+1}/{N}] done")
    return {
        "contrib_real_means":     torch.stack(contribs_real).numpy(),
        "contrib_shuffled_means": torch.stack(contribs_shuf).numpy(),
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


def svd_out(enc: RenderSpatialEncoder) -> Dict[str, np.ndarray]:
    """SVD of egowm render_encoder.out.weight (out_dim, hidden_dim)."""
    W = enc.out.weight.detach().float().cpu().numpy()
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
    ax.set_title(f"render_encoder.out.weight SVD\n(eff_rank={svd['eff_rank'][0]:.2f} / {svd['S'].shape[0]})")
    ax.set_xlabel("singular index"); ax.set_ylabel("σ (log)")
    ax.grid(alpha=0.3)

    # 2. contribution per-scene cos-sim heatmap
    ax = axes[0, 1]
    im = ax.imshow(contrib_cos, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(f"contribution per-scene cos-sim\n(mean off-diag={summary['contrib_cos_offdiag_mean']:.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3. (egowm has no cross-attn V — placeholder for layout parity with v2)
    ax = axes[0, 2]; ax.axis("off")
    ax.text(0.5, 0.5, "egowm has no cross-attn V\n(probe n/a)",
            ha="center", va="center", fontsize=11, family="monospace")

    # 4. PCA scatter of per-scene contribution means
    ax = axes[1, 0]
    c = contrib_means - contrib_means.mean(axis=0, keepdims=True)
    Uc, Sc, _ = np.linalg.svd(c, full_matrices=False)
    proj = Uc[:, :2] * Sc[:2]
    ax.scatter(proj[:, 0], proj[:, 1], s=40)
    for i in range(proj.shape[0]):
        ax.annotate(str(i), (proj[i, 0], proj[i, 1]), fontsize=7)
    ax.set_title(f"PCA of per-scene contribution means\n(top-2 var = {(Sc[:2]**2 / (Sc**2).sum()).sum()*100:.1f}%)")
    ax.set_xlabel(f"PC1 ({Sc[0]**2 / (Sc**2).sum()*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({Sc[1]**2 / (Sc**2).sum()*100:.1f}%)")
    ax.grid(alpha=0.3)

    # 5. Render-shuffle delta histogram
    ax = axes[1, 1]
    ax.hist(shuffle_deltas, bins=20, edgecolor="black")
    ax.axvline(summary["shuffle_delta_median"], color="red", ls="--",
               label=f"median={summary['shuffle_delta_median']:.3f}")
    ax.set_title("Render-shuffle: ||c_real − c_shuf|| / ||c_real||\n(0 = render ignored, 1 = render decisive)")
    ax.set_xlabel("relative delta"); ax.legend(); ax.grid(alpha=0.3)

    # 6. Summary text panel
    ax = axes[1, 2]; ax.axis("off")
    lines = [
        "── egowm collapse diagnostic ──",
        "",
        f"effective rank of render_encoder.out.weight: {summary['out_eff_rank']:.2f} / {svd['S'].shape[0]}",
        f"top-1 singular value share:                  {summary['out_sigma1_share']*100:.1f}%",
        "",
        f"R-ratio (shared / per-scene):                {summary['R_ratio']:.2f}",
        f"contrib per-scene cos (off-diag):            {summary['contrib_cos_offdiag_mean']:.3f}",
        "",
        "render-shuffle (real vs other scene's render):",
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

    print("=== egowm collapse diagnostic ===")
    print(f"ckpt:    {args.ckpt}")
    print(f"eval:    {args.eval_csv}  n_scenes={args.n_scenes}")
    print(f"device:  {device}  dtype={dtype}")

    enc = load_egowm_encoder(args, device, dtype)
    cache = build_eval_cache(args, device, dtype)
    if len(cache) < 2:
        raise RuntimeError(f"need at least 2 scenes; got {len(cache)}")

    svd = svd_out(enc)
    stats = per_scene_stats(enc, cache, device, dtype)

    c_cos = cos_sim_matrix(stats["contrib_real_means"])
    N = c_cos.shape[0]
    offdiag = ~np.eye(N, dtype=bool)
    R, shared_mag, per_scene_resid_mag = r_ratio(stats["contrib_real_means"])

    c_real = stats["contrib_real_means"]
    c_shuf = stats["contrib_shuffled_means"]
    diff = np.linalg.norm(c_real - c_shuf, axis=1)
    base = np.linalg.norm(c_real, axis=1) + 1e-12
    shuffle_deltas = diff / base

    sigma1_share = float(svd["S"][0] ** 2 / max((svd["S"] ** 2).sum(), 1e-12))

    summary: Dict[str, float] = {
        "n_scenes":                int(N),
        "out_eff_rank":            float(svd["eff_rank"][0]),
        "out_sigma1_share":        sigma1_share,
        "contrib_cos_offdiag_mean":float(c_cos[offdiag].mean()),
        "R_ratio":                 float(R),
        "shared_mag":              float(shared_mag),
        "per_scene_resid_mag":     float(per_scene_resid_mag),
        "shuffle_delta_mean":      float(shuffle_deltas.mean()),
        "shuffle_delta_median":    float(np.median(shuffle_deltas)),
        "shuffle_delta_p90":       float(np.quantile(shuffle_deltas, 0.9)),
        "ckpt":                    str(args.ckpt),
    }

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        out_dir / "raw.npz",
        contrib_real_means=stats["contrib_real_means"],
        contrib_shuffled_means=stats["contrib_shuffled_means"],
        out_S=svd["S"],
        out_U_top=svd["U_top"],
        out_Vh_top=svd["Vh_top"],
        contrib_cos=c_cos,
        shuffle_deltas=shuffle_deltas,
    )
    make_figure(
        out_dir / "diag.png",
        svd, c_cos,
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
