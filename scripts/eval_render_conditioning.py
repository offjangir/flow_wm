#!/usr/bin/env python3
"""
Held-out evaluation of a render-conditioned Wan I2V checkpoint.

Single-GPU. Loads a training config + a saved ``render_conditioner.pt`` (the
subset checkpoint train_fsdp.py writes), runs the same precompute pipeline on
a held-out test CSV, then reports:

  - **flow loss on held-out** : flow MSE averaged over the held-out samples
                                (over many random timesteps each)
  - **condition_usage_sanity**: for each sample, compare flow loss with the
                                CORRECT render vs a SHUFFLED render. If the
                                wrong-render loss is meaningfully higher, the
                                model is using render conditioning.
                                  ``loss_gap``      = wrong - right
                                  ``loss_gap_ratio``= gap / right (relative)
                                  ``wrong_higher`` = fraction of samples
                                                     where wrong > right
                                A healthy run shows positive gap + frac > 0.5.

Usage::

  python scripts/eval_render_conditioning.py \\
      --config       configs/train_drrobot_1k_legacy_8xl40.json \\
      --ckpt         checkpoints/wan_render_drrobot_1k_legacy_8xl40/run_.../epoch_4/render_conditioner.pt \\
      --eval_csv     data_wan_eval/eval_metadata.csv \\
      --dataset_base_path data_wan_eval \\
      --ignore_prompts \\
      --num_timesteps_per_sample 8 \\
      --device cuda:0

  If neither ``--ignore_prompts`` nor ``--use_dataset_prompts`` is passed,
  behaviour follows the JSON ``ignore_prompts`` field.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _p in (SRC, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from world_model.wan_flow.data import RenderI2VMetadataDataset            # noqa: E402
from world_model.wan_flow.train import (                                   # noqa: E402
    _materialize_meta_submodules,
    _precompute_embeddings,
    condition_usage_sanity,
)
from world_model.wan_flow.train_fsdp import _load_dit_only                 # noqa: E402
from _eval_common import verify_eval_matches_training                       # noqa: E402


def _load_render_conditioner_subset(model: torch.nn.Module, ckpt_path: str) -> None:
    """The training run saves ONLY the trainable subset of params (render
    conditioner + tracks_head + last N unfrozen blocks). Load these with
    ``strict=False`` so the rest of Wan stays at its pretrained values."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Some checkpoints save fp32 tensors; cast to whatever the live model uses.
    target_dtype = next(model.parameters()).dtype
    sd_cast = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            sd_cast[k] = v.to(dtype=target_dtype)
        else:
            sd_cast[k] = v
    missing, unexpected = model.load_state_dict(sd_cast, strict=False)
    n_loaded = len(sd_cast) - len(unexpected)
    print(f"[ckpt] loaded {n_loaded}/{len(sd_cast)} tensors from {ckpt_path}")
    if unexpected:
        print(f"[ckpt] WARN unexpected keys (first 5): {unexpected[:5]}")
    # `missing` is huge here (everything we DIDN'T train) — no need to print.


@torch.no_grad()
def eval_flow_loss(
    model: torch.nn.Module,
    embed_cache: list,
    scheduler,
    device: torch.device,
    dtype: torch.dtype,
    num_timesteps_per_sample: int,
    seed: int,
) -> dict:
    """Held-out flow MSE, averaged over (samples × random timesteps).

    Bucketed by timestep quartile so we can see whether the model has learned
    the harder mid-sigma regions, not just the trivial near-clean / near-noise
    ends.
    """
    n_t = len(scheduler.timesteps)
    bucket_edges = [n_t * k // 4 for k in range(5)]   # [0, n/4, n/2, 3n/4, n]
    bucket_sum = [0.0, 0.0, 0.0, 0.0]
    bucket_cnt = [0, 0, 0, 0]
    all_losses = []

    g = torch.Generator()
    g.manual_seed(int(seed))

    for i, c in enumerate(embed_cache):
        clean = c["clean_latents"].to(device=device, dtype=dtype, non_blocking=True)
        cond  = c["condition"].to(device=device, dtype=dtype, non_blocking=True)
        prompt = c["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        image  = c["image_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        rr     = c["render_latents"].to(device=device, dtype=dtype, non_blocking=True)

        for _ in range(num_timesteps_per_sample):
            noise = torch.randn(clean.shape, generator=g).to(device=device, dtype=dtype)
            t_idx = int(torch.randint(0, n_t, (1,), generator=g).item())
            t_scalar = scheduler.timesteps[t_idx].to(device=device).float()
            t_batch  = t_scalar.expand(clean.shape[0])
            noisy = scheduler.scale_noise(clean, t_batch, noise)
            noisy[:, :, 0:1] = clean[:, :, 0:1]
            latent_in = torch.cat([noisy, cond], dim=1)
            target = noise - clean

            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                pred = model(
                    hidden_states=latent_in,
                    timestep=t_batch,
                    encoder_hidden_states=prompt,
                    encoder_hidden_states_image=image,
                    render_latents=rr,
                    return_dict=False,
                )[0]

            loss = float(F.mse_loss(pred[:, :, 1:].float(), target[:, :, 1:].float()))
            all_losses.append(loss)
            b = min(3, max(0, sum(1 for e in bucket_edges[1:] if t_idx >= e)))
            bucket_sum[b] += loss
            bucket_cnt[b] += 1

    mean_all = sum(all_losses) / max(1, len(all_losses))
    out = {"flow_loss_mean": mean_all, "n_samples": len(embed_cache),
           "n_evals": len(all_losses)}
    for k in range(4):
        if bucket_cnt[k] > 0:
            out[f"flow_loss_q{k}"] = bucket_sum[k] / bucket_cnt[k]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Training config JSON to inherit settings from.")
    p.add_argument("--ckpt",   required=True, help="Path to render_conditioner.pt (the trainable subset).")
    p.add_argument("--eval_csv", required=True, help="Held-out metadata CSV (default: 10-video subset).")
    p.add_argument(
        "--dataset_base_path",
        default=None,
        help="Directory relative to which CSV paths (video/render/tracks) are resolved. "
        "Training JSON usually has data_wan_1k; held-out CSVs from build_eval_set.sh live under "
        "data_wan_eval — pass data_wan_eval here. Omit to use the JSON dataset_base_path.",
    )
    p.add_argument("--num_timesteps_per_sample", type=int, default=8,
                   help="Random timesteps to average over per sample (denoises the eval signal).")
    p.add_argument("--num_sanity_pairs", type=int, default=10,
                   help="Number of (right vs shuffled-render) probe pairs.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", type=str, default=None,
                   help="Optional path to write a JSON results file alongside stdout.")
    prompt_g = p.add_mutually_exclusive_group()
    prompt_g.add_argument(
        "--ignore_prompts",
        action="store_true",
        help="Encode empty strings for text (matches train_fsdp --ignore_prompts). Overrides JSON.",
    )
    prompt_g.add_argument(
        "--use_dataset_prompts",
        action="store_true",
        help="Use the CSV ``prompt`` column. Overrides JSON ``ignore_prompts``.",
    )
    p.add_argument(
        "--no_strict_args", action="store_true",
        help="Skip the train/eval arg-mismatch check. Default: error on any "
             "critical mismatch (model_path, num_frames, embodiment_kwargs, ...).",
    )
    args = p.parse_args()

    cfg = json.load(open(args.config))
    # Sanity: this checkpoint must have been trained with the same architecture
    # / shapes / dtype the eval is using. Otherwise load_state_dict silently
    # mismatches and we evaluate on partially-loaded weights.
    verify_eval_matches_training(cfg, args.ckpt, strict=not args.no_strict_args)

    device = torch.device(args.device)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[
        cfg.get("mixed_precision", "bf16")
    ]

    if args.ignore_prompts:
        ignore_prompts = True
    elif args.use_dataset_prompts:
        ignore_prompts = False
    else:
        ignore_prompts = bool(cfg.get("ignore_prompts", False))
    src = "CLI" if (args.ignore_prompts or args.use_dataset_prompts) else "config"
    print(f"[eval] ignore_prompts={ignore_prompts} (from {src})")

    ds_root = args.dataset_base_path if args.dataset_base_path is not None else cfg["dataset_base_path"]
    # ---- 1. dataset over the eval CSV ----
    print(f"[eval] dataset: {args.eval_csv}  |  base_path={ds_root}")
    eval_dataset = RenderI2VMetadataDataset(
        base_path=ds_root,
        metadata_csv=args.eval_csv,
        num_frames=cfg["num_frames"],
        height=cfg["height"],
        width=cfg["width"],
        repeat=1,
    )
    print(f"[eval] {len(eval_dataset.rows)} held-out samples (frames={cfg['num_frames']}, "
          f"{cfg['height']}x{cfg['width']})")

    # ---- 2. precompute eval embeddings (no on-disk cache; eval set is small) ----
    print(f"[eval] precompute (text/image/VAE encoders one at a time)…")
    embed_cache, _z_dim, scheduler = _precompute_embeddings(
        model_path=cfg["model_path"],
        dataset=eval_dataset,
        device=device,
        height=cfg["height"],
        width=cfg["width"],
        num_frames=cfg["num_frames"],
        max_seq_len=cfg.get("max_sequence_length", 256),
        encoder_dtype=dtype,
        ignore_prompts=ignore_prompts,
        cache_path=None,
        safe_loading=True,
    )
    print(f"[eval] cached {len(embed_cache)} samples")

    # ---- 3. build DiT, load checkpoint subset ----
    print(f"[eval] building DiT and loading subset checkpoint…")
    dit = _load_dit_only(
        cfg["model_path"], dtype,
        use_embodiment_adapter=bool(cfg.get("use_embodiment_adapter", False)),
        embodiment_kwargs=cfg.get("embodiment_kwargs"),
        legacy_render_variant=cfg.get("legacy_render_variant", "egowm"),
    )
    _materialize_meta_submodules(dit)
    if hasattr(dit, "reset_zero_gates"):
        dit.reset_zero_gates()
    _load_render_conditioner_subset(dit, args.ckpt)
    dit = dit.to(device=device, dtype=dtype).eval()

    # ---- 4. flow loss on held-out ----
    print(f"\n[eval] flow loss on {len(embed_cache)} held-out samples × "
          f"{args.num_timesteps_per_sample} random timesteps each…")
    flow_stats = eval_flow_loss(
        dit, embed_cache, scheduler, device, dtype,
        num_timesteps_per_sample=args.num_timesteps_per_sample,
        seed=args.seed,
    )

    # ---- 5. condition_usage_sanity (right vs shuffled render) ----
    print(f"[eval] condition_usage_sanity over {args.num_sanity_pairs} sample pairs…")
    sanity = condition_usage_sanity(
        dit, embed_cache, scheduler, device, dtype,
        num_samples=args.num_sanity_pairs,
        seed=args.seed,
    )

    # ---- 6. report ----
    print("\n" + "=" * 72)
    print("FLOW LOSS (held-out)")
    print("=" * 72)
    print(f"  mean over {flow_stats['n_evals']} (sample × timestep) evals: "
          f"{flow_stats['flow_loss_mean']:.4f}")
    for k in range(4):
        key = f"flow_loss_q{k}"
        if key in flow_stats:
            print(f"  q{k} (sigma quartile {k}/3): {flow_stats[key]:.4f}")
    print()
    print("=" * 72)
    print("CONDITION USAGE SANITY  (correct render vs shuffled render)")
    print("=" * 72)
    if sanity is None:
        print("  (skipped: not enough samples for shuffle pairing)")
    else:
        print(f"  samples            : {int(sanity['samples'])}")
        print(f"  loss right (correct): {sanity['loss_right']:.4f}")
        print(f"  loss wrong (shuffled): {sanity['loss_wrong']:.4f}")
        print(f"  loss gap (wrong - right): {sanity['loss_gap']:+.4f}  "
              f"({100.0 * sanity['loss_gap_ratio']:+.1f}%)")
        print(f"  wrong > right on: {100.0 * sanity['wrong_higher_frac']:.1f}% of samples")
        print()
        if sanity["loss_gap"] <= 0.0:
            print("  ✗ Model is NOT distinguishing the correct render. "
                  "Conditioning is not effectively wired into prediction.")
        elif sanity["loss_gap_ratio"] < 0.05:
            print("  △ Marginal: gap < 5% of right loss. Conditioning is being "
                  "used weakly — gates may need more training.")
        else:
            print("  ✓ Model uses render conditioning meaningfully.")

    # ---- 7. optionally write JSON ----
    if args.out_json:
        out = {
            "config": args.config,
            "ckpt": args.ckpt,
            "eval_csv": args.eval_csv,
            "n_samples": len(embed_cache),
            "flow_stats": flow_stats,
            "sanity": sanity,
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[eval] wrote results JSON → {args.out_json}")


if __name__ == "__main__":
    main()
