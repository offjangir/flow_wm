
"""
Multi-GPU FSDP training for render-conditioned Wan 2.1 I2V.

Shards the 14B DiT across GPUs via Fully Sharded Data Parallel so each GPU
only holds a fraction of the model weights + optimizer states.

Run::

  accelerate launch --config_file configs/accelerate_fsdp.yaml \
    -m world_model.wan_flow.train_fsdp --config configs/train_drrobot_fsdp.json

  # Or explicitly:
  accelerate launch --num_processes 4 --config_file configs/accelerate_fsdp.yaml \
    -m world_model.wan_flow.train_fsdp \
    --model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --dataset_base_path ./data_wan \
    --metadata_csv ./data_wan/metadata.csv

Continue training from a saved ``render_conditioner.pt`` (new ``run_*`` dir;
checkpoints named ``epoch_{offset+inner}``)::

  accelerate launch --config_file configs/accelerate_fsdp.yaml \
    -m world_model.wan_flow.train_fsdp \
    --config configs/train_drrobot_1k_legacy_8xl40.json \
    --resume_render_ckpt checkpoints/.../run_.../epoch_29/render_conditioner.pt \
    --epoch_offset 30 --num_epochs 30 \
    --wandb_run_name my_run_continue30
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from world_model.wan_flow.data import RenderI2VMetadataDataset
from world_model.wan_flow.model import (
    WanTransformerRenderConditioned,
)
from world_model.wan_flow.train import (
    _PRECOMPUTE_CACHE_VERSION,
    _compute_precompute_key,
    _precompute_embeddings,
    _setup_run_dir_and_logging,
    build_lr_scheduler,
    condition_usage_sanity,
    fsdp_micro_steps_per_epoch,
    optimizer_step_count,
    save_forward_debug_bundle,
)


def _precompute_embeddings_sharded(
    accelerator,
    model_path,
    dataset,
    height,
    width,
    num_frames,
    max_seq_len,
    encoder_dtype,
    ignore_prompts: bool,
    cache_path,
):
    """
    Distributed precompute: each rank encodes a contiguous slice of the
    dataset on its own GPU, then results are all-gathered to construct the
    full cache.  Cuts wall time ~world_size × vs. the rank-0-only path.

    Why this exists: the original rank-0-only implementation was forced by a
    `from_pretrained(device_map="cuda:N")` quirk under
    ``fsdp_cpu_ram_efficient_loading=true`` that drops weights to CPU on
    non-rank-0 processes.  We sidestep it by passing ``safe_loading=True`` to
    the inner function, which uses the plain ``.from_pretrained(...).to(device)``
    path on every rank.

    The on-disk cache key is computed from the FULL row list (so it remains
    a single shared cache file), but the encode work is parallelised.
    """
    import os as _os
    import torch.distributed as dist
    from pathlib import Path as _Path

    rank = accelerator.process_index
    world = accelerator.num_processes
    rows = dataset.rows
    n = len(rows)
    cache_path = _Path(cache_path) if cache_path else None

    # ── 1. Cache hit fast-path: every rank loads the full cache from disk ──
    cache_key = None
    if cache_path is not None:
        cache_key = _compute_precompute_key(
            model_path=str(model_path).rstrip("/"),
            rows=rows,
            base_path=dataset.base_path,
            num_frames=num_frames,
            height=height,
            width=width,
            max_seq_len=max_seq_len,
            ignore_prompts=ignore_prompts,
            has_tracks=dataset.has_tracks,
            has_actions=getattr(dataset, "has_actions", False),
            action_field=getattr(dataset, "action_field", "state"),
            action_stats_path=getattr(dataset, "_action_stats_path", "") or "",
        )
        if cache_path.is_file():
            try:
                blob = torch.load(cache_path, map_location="cpu", weights_only=False)
                if blob.get("key") == cache_key:
                    if accelerator.is_main_process:
                        accelerator.print(
                            f"[precompute-sharded] HIT  ({len(blob['cache'])} samples, "
                            f"{cache_path.stat().st_size / 1e9:.2f} GB)  → skipping encoders"
                        )
                    accelerator.wait_for_everyone()
                    return blob["cache"], blob["z_dim"], blob["scheduler"]
                if accelerator.is_main_process:
                    accelerator.print(
                        f"[precompute-sharded] MISS — cache key changed "
                        f"(stored={blob.get('key', '?')[:16]}…  expected={cache_key[:16]}…)"
                    )
            except Exception as exc:  # noqa: BLE001
                if accelerator.is_main_process:
                    accelerator.print(
                        f"[precompute-sharded] cache load failed ({exc}); recomputing"
                    )

    # ── 2. Cache miss: shard rows contiguously across ranks ────────────────
    chunk = (n + world - 1) // world
    start = rank * chunk
    end = min(start + chunk, n)
    my_rows = rows[start:end]

    if accelerator.is_main_process:
        accelerator.print(
            f"[precompute-sharded] {n} samples → {world} ranks, ~{chunk}/rank "
            f"(rank 0 does rows[{start}:{end}])"
        )
    accelerator.print(
        f"[rank {rank}] encoding rows[{start}:{end}] ({len(my_rows)} samples)"
    )

    my_cache, z_dim, scheduler = _precompute_embeddings(
        model_path=model_path,
        dataset=dataset,
        device=accelerator.device,
        height=height,
        width=width,
        num_frames=num_frames,
        max_seq_len=max_seq_len,
        encoder_dtype=encoder_dtype,
        ignore_prompts=ignore_prompts,
        cache_path=None,
        rows=my_rows,
        safe_loading=True,
        save_cache=False,
    )

    # ── 3. Gather slices → full cache, in original row order ──────────────
    gathered: list = [None] * world
    dist.all_gather_object(gathered, my_cache)
    full_cache: list = []
    for shard in gathered:
        full_cache.extend(shard)
    if len(full_cache) != n:
        raise RuntimeError(
            f"[precompute-sharded] gather size mismatch: got {len(full_cache)}, expected {n}"
        )

    # ── 4. Rank 0 saves to disk ────────────────────────────────────────────
    if accelerator.is_main_process and cache_path is not None and cache_key is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            torch.save(
                {
                    "key": cache_key,
                    "cache": full_cache,
                    "z_dim": z_dim,
                    "scheduler": scheduler,
                    "version": _PRECOMPUTE_CACHE_VERSION,
                },
                tmp,
            )
            _os.replace(tmp, cache_path)
            accelerator.print(
                f"[precompute-sharded] saved cache → {cache_path} "
                f"({cache_path.stat().st_size / 1e9:.2f} GB)"
            )
        except Exception as exc:  # noqa: BLE001
            accelerator.print(
                f"[precompute-sharded] WARN: failed to write cache to {cache_path}: {exc}"
            )
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    accelerator.wait_for_everyone()
    return full_cache, z_dim, scheduler


def _cast_module_for_fsdp(module: torch.nn.Module, dtype: torch.dtype) -> Dict[str, int]:
    """
    FSDP flattening requires a uniform floating dtype within each wrapped module.
    The render-conditioned Wan DiT keeps a few modules in fp32 for stability, which
    can trip FSDP at `accelerator.prepare(...)` when the rest of the model is bf16/fp16.
    Cast all floating params/buffers to the requested dtype before wrapping.
    """
    n_params = 0
    n_buffers = 0
    for p in module.parameters():
        if p.is_floating_point() and p.dtype != dtype:
            p.data = p.data.to(dtype=dtype)
            n_params += p.numel()
    for b in module.buffers():
        if b.is_floating_point() and b.dtype != dtype:
            b.data = b.data.to(dtype=dtype)
            n_buffers += b.numel()
    return {"params": n_params, "buffers": n_buffers}


def _materialize_meta_submodules(model: torch.nn.Module) -> int:
    """
    Replace meta tensors (typically from newly added modules absent in checkpoint)
    with real CPU tensors before any `.cpu()` / FSDP wrapping.
    """
    meta_modules = [
        m
        for _name, m in model.named_modules()
        if any(p.is_meta for p in m.parameters(recurse=False))
        or any(b.is_meta for b in m.buffers(recurse=False))
    ]
    if not meta_modules:
        return 0

    for m in meta_modules:
        m.to_empty(device="cpu", recurse=False)
        # PyTorch convention is mixed: nn.MultiheadAttention only exposes the
        # private `_reset_parameters`, not the public `reset_parameters`. Without
        # the elif below we'd fall through to the zero-everything branch and
        # silently zero MHA's `in_proj_weight` → cross-attn outputs zero →
        # adapter gates never get gradient. See model.py:reset_zero_gates for
        # the symptom-level workaround that this elif obviates.
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()
        elif hasattr(m, "_reset_parameters") and callable(m._reset_parameters):
            m._reset_parameters()
        else:
            for p in m.parameters(recurse=False):
                with torch.no_grad():
                    p.zero_()
            for b in m.buffers(recurse=False):
                with torch.no_grad():
                    b.zero_()
    return len(meta_modules)


@torch.no_grad()
def _load_resume_render_ckpt(
    dit: torch.nn.Module, ckpt_path: str, accelerator: Accelerator
) -> None:
    """Load a saved ``render_conditioner.pt`` (trainable subset) before FSDP.

    Same layout as inference: ``strict=False`` so only overlapping keys move.
    All ranks load identically from the shared filesystem.
    """
    path = ckpt_path.strip()
    sd = torch.load(path, map_location="cpu", weights_only=False)
    target_dtype = next(dit.parameters()).dtype
    sd_cast = {
        k: (
            v.to(dtype=target_dtype)
            if isinstance(v, torch.Tensor) and v.is_floating_point()
            else v
        )
        for k, v in sd.items()
    }
    missing, unexpected = dit.load_state_dict(sd_cast, strict=False)
    if accelerator.is_main_process:
        n_ok = len(sd_cast) - len(unexpected)
        accelerator.print(
            f"[FSDP train] resume: loaded {n_ok}/{len(sd_cast)} tensors from {path}"
        )
        if unexpected:
            accelerator.print(
                f"[FSDP train] resume WARN unexpected keys (first 8): {unexpected[:8]}"
            )
        if missing:
            accelerator.print(
                f"[FSDP train] resume WARN missing keys (first 8): {missing[:8]}"
            )


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre.parse_known_args()
    defaults: Dict[str, Any] = {}
    if pre_args.config:
        with open(pre_args.config, encoding="utf-8") as f:
            defaults = json.load(f)

    p = argparse.ArgumentParser(
        description="Multi-GPU FSDP training for render-conditioned Wan 2.1 I2V.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model_path", type=str, required=False)
    p.add_argument("--dataset_base_path", type=str, required=False)
    p.add_argument("--metadata_csv", type=str, required=False)
    p.add_argument(
        "--action_stats_path", type=str, default=None,
        help="JSON with per-dim mean/std for z-score normalizing the action "
             "stream. When set, the dataset returns normalized 'actions' "
             "(num_frames, D) and the precompute cache stores them.")
    p.add_argument(
        "--action_field", type=str, default="state",
        help="Which field in actions/<scene>.npz to use ('state' = 8-d "
             "joint+gripper; 'action' = 7-d cmd-target+gripper).")
    p.add_argument("--output_dir", type=str, default="./checkpoints/wan_render_fsdp")
    p.add_argument(
        "--precompute_cache_path", type=str, default=None,
        help="Override location of the precomputed embed cache (.pt). When set, "
             "the cache is read/written here instead of `<output_dir>/embed_cache.pt`. "
             "Lets multiple training configs share one cache (e.g. main + smoke), "
             "since the cache contents are keyed by data + resolution + num_frames "
             "+ ignore_prompts, not by output_dir.",
    )
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--dataset_repeat", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument(
        "--gate_lr_multiplier", type=float, default=10.0,
        help="LR multiplier for zero-init params (gates + tracks_head's final "
             "layer). Default 10.0 (right for embodiment adapter where many "
             "gates need to escape exactly-zero init). For the EgoWM-style "
             "legacy variant (no gates), 1.0-3.0 is typically appropriate.",
    )
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_warmup",
        choices=("constant", "cosine", "cosine_warmup", "cosine_epoch", "epoch_linear", "warmup_epoch_linear"),
        help="constant: fixed; cosine: cosine to min over full run, then hold; "
        "cosine_warmup: warmup+that; cosine_epoch: hold, cosine in lr_epoch_*, then hold; "
        "epoch_linear: linear in window; warmup_epoch_linear: warmup+linear window.",
    )
    p.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=300,
        help="Warmup steps (0 = none). Ignored for constant, cosine, and epoch_linear.",
    )
    p.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.1,
        help="For cosine: eta_min. For warmup_epoch_linear: final = lr*this (e.g. 0.01 → 1e-6 with lr=1e-4).",
    )
    p.add_argument(
        "--lr_epoch_decay_start",
        type=int,
        default=300,
        help="(cosine_epoch / epoch_linear / warmup_epoch_linear) Decay window start epoch (0-based).",
    )
    p.add_argument(
        "--lr_epoch_decay_end",
        type=int,
        default=800,
        help="(cosine_epoch / epoch_linear / warmup_epoch_linear) Floor LR at first step of this epoch, then hold.",
    )
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument(
        "--resume_render_ckpt",
        type=str,
        default=None,
        help="Path to render_conditioner.pt from a finished run. Loads the "
        "trainable subset into the DiT (strict=False) on CPU before FSDP. "
        "Requires --epoch_offset > 0 so new epoch_* dirs do not overwrite "
        "the previous run's checkpoints.",
    )
    p.add_argument(
        "--epoch_offset",
        type=int,
        default=0,
        help="Added to the inner epoch index for RNG seeds, logs, and "
        "checkpoint folder names (epoch_{offset + inner}). Use 30 after "
        "completing epoch_29 when continuing training.",
    )
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--trainable", type=str, default="render_only",
        choices=("render_only", "full_dit"),
    )
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=("no", "fp16", "bf16"))
    p.add_argument("--unfreeze_last_n_blocks", type=int, default=0, help="Number of final DiT blocks to unfreeze in render_only mode")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_every_n_epochs", type=int, default=10)
    p.add_argument("--log_every_n_steps", type=int, default=5)
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Gradient clipping norm; also used as the NaN/Inf guard threshold")
    p.add_argument("--abort_on_nan", action="store_true",
                   help="If set, raise on first non-finite loss/grad instead of skipping the step")
    p.add_argument(
        "--condition_usage_sanity_samples",
        type=int,
        default=0,
        help="If > 0, at save-time compare flow loss with correct render vs shuffled render "
             "for this many cached samples (conditioning-usage sanity probe).",
    )
    p.add_argument(
        "--disable_render_gate",
        action="store_true",
        help="If set, force render_gate=1.0 and freeze it (no attenuation, no gate learning).",
    )
    p.add_argument(
        "--forward_debug_dir",
        type=str,
        default=None,
        help="If set, on the matching (epoch, step) on the main process, save tensors used in forward "
             "+ flow loss (see save_forward_debug_bundle in train.py). Relative → under output_dir.",
    )
    p.add_argument("--forward_debug_epoch", type=int, default=0)
    p.add_argument("--forward_debug_step_in_epoch", type=int, default=0)
    p.add_argument("--forward_debug_fps", type=int, default=8)
    p.add_argument(
        "--forward_debug_rgb_from_disk",
        action="store_true",
        help="Also write optional_disk_* RGB files by re-reading CSV paths (not the VAE tensors).",
    )
    # ---- auxiliary tracks-head supervision ----
    p.add_argument("--lambda_tracks", type=float, default=0.1)
    p.add_argument("--ref_frame", type=int, default=0)
    p.add_argument("--max_query_points", type=int, default=-1)
    # ---- in-training eval (every epoch, no VAE decode + at save epochs, VAE decode) ----
    p.add_argument(
        "--eval_every_n_epochs", type=int, default=0,
        help="Run held-out flow-MSE eval every N epochs (0 disables all "
             "in-training eval). Cheap: ~30 s per call.",
    )
    p.add_argument(
        "--eval_csv_heldout", type=str, default=None,
        help="Held-out metadata CSV for eval. Defaults to <output_dir>/" +
             "../train_metadata_test.csv heuristic.",
    )
    p.add_argument("--eval_n_heldout_flow", type=int, default=5,
                   help="Number of held-out samples for flow-MSE eval.")
    p.add_argument("--eval_timesteps_per_sample", type=int, default=4,
                   help="Random timesteps per held-out sample for flow-MSE.")
    p.add_argument(
        "--eval_video_at_save", action="store_true",
        help="At every save epoch (where a render_conditioner.pt is dumped), "
             "also run a small video eval with PSNR/SSIM. Re-loads VAE on "
             "rank 0 temporarily.",
    )
    p.add_argument("--eval_n_heldout_video", type=int, default=3,
                   help="Number of held-out samples for video PSNR/SSIM eval.")
    p.add_argument("--eval_n_train_video", type=int, default=2,
                   help="Number of train samples for video PSNR/SSIM eval "
                        "(overfit-gap signal).")
    p.add_argument("--eval_inference_steps", type=int, default=30)
    p.add_argument("--eval_cfg_scale", type=float, default=1.0)
    # ---- wandb ----
    p.add_argument("--wandb_project", type=str, default="wan_flow_drrobot")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=("online", "offline", "disabled"))
    p.add_argument(
        "--drop_render_conditioning",
        action="store_true",
        help="If set, pass render_latents=None to the model to run vanilla I2V DiT behavior.",
    )
    p.add_argument(
        "--render_dropout_prob",
        type=float,
        default=0.0,
        help="Per-step probability of dropping the render condition during training "
             "(render_latents=None). Required for inference-time CFG-on-render to be "
             "meaningful: with prob=0 the model never sees a no-render input and the "
             "unconditional CFG branch is out-of-distribution. 0.1 is a standard value.",
    )
    p.add_argument(
        "--use_embodiment_adapter",
        action="store_true",
        help="If set, replace the legacy pooled-vector render encoder + scalar gate "
             "(EgoWM Eq. 5) with the EmbodimentAgnosticConditioning module: spatial KV "
             "bank + per-block cross-attn adapters + decoupled AdaLN deltas. Identity "
             "at init. See ARCHITECTURE_REVIEW.md §4.1.",
    )
    p.add_argument(
        "--legacy_render_variant",
        type=str,
        default="egowm",
        choices=("egowm", "v1", "v2", "v3"),
        help="Selects which legacy (non-embodiment-adapter) render-conditioning "
             "design to use:\n"
             "  egowm (default): per-token spatial render encoder added "
             "directly to per-token temb, no gate, no fuse Linear. Mirrors "
             "EgoWM Eq. 5–6. No actions.\n"
             "  v1 (BACKWARD-COMPAT): pooled-per-frame render encoder + Linear "
             "fuse + scalar render_gate(=0.1). Use to load existing v1 ckpts.\n"
             "  v2 (action-conditioned, cross-attn): render Q ← cross_attn → "
             "actions K/V → LayerNorm → zero-init Linear → ADD to temb_base. "
             "Suffers V-span attractor (universal-tint bug).\n"
             "  v3 (action-conditioned, concat): per-(t, h, w) concat of "
             "render and broadcast action features → MLP → zero-init Linear "
             "→ ADD to temb_base. Avoids v2's bias-direction trap because "
             "render's per-token diversity is preserved (no softmax mixing).",
    )
    p.add_argument(
        "--embodiment_kwargs",
        type=lambda s: json.loads(s) if isinstance(s, str) else s,
        default=None,
        help="JSON dict of kwargs forwarded to EmbodimentAgnosticConditioning "
             "(e.g. {\"use_action_aware_adaln\": true, "
             "\"action_aware_kwargs\": {\"spatial_pool\": 2, \"action_dim\": null}}). "
             "Pass via --config JSON; CLI accepts a JSON string.",
    )
    p.add_argument(
        "--v2_adapter_kwargs",
        type=lambda s: json.loads(s) if isinstance(s, str) else s,
        default=None,
        help="JSON dict of kwargs forwarded to ActionConditionedTembAdapter "
             "when --legacy_render_variant=v2 (e.g. "
             "{\"action_dim\": 8, \"hidden_dim\": 512, \"num_heads\": 8, "
             "\"spatial_downsample\": 2}). action_dim is required.",
    )
    p.add_argument(
        "--v3_adapter_kwargs",
        type=lambda s: json.loads(s) if isinstance(s, str) else s,
        default=None,
        help="JSON dict of kwargs forwarded to ActionRenderConcatAdapter "
             "when --legacy_render_variant=v3 (e.g. "
             "{\"action_dim\": 8, \"hidden_dim\": 512, \"spatial_downsample\": 2, "
             "\"action_temporal_align\": \"avg_pool\"}). action_dim is required.",
    )
    p.add_argument(
        "--ignore_prompts",
        action="store_true",
        help="If set, use empty string for all text prompts during precompute.",
    )
    p.set_defaults(**{k: v for k, v in defaults.items() if k != "config"})
    args = p.parse_args(remaining)
    if not args.model_path or not args.dataset_base_path or not args.metadata_csv:
        p.error("Provide --model_path, --dataset_base_path, and --metadata_csv (via CLI or --config JSON).")
    return args


def _load_dit_only(
    model_path: str,
    dtype: torch.dtype,
    use_embodiment_adapter: bool = False,
    embodiment_kwargs: Optional[Dict[str, Any]] = None,
    legacy_render_variant: str = "egowm",
    v2_adapter_kwargs: Optional[Dict[str, Any]] = None,
    v3_adapter_kwargs: Optional[Dict[str, Any]] = None,
) -> WanTransformerRenderConditioned:
    """
    Load *only* the render-conditioned DiT (no VAE / text / image encoders).
    Encoders are run once at startup by ``_precompute_embeddings`` and then
    freed; they are never on the GPU during the FSDP training loop.
    """
    root = model_path.rstrip("/")
    local_transformer = os.path.isdir(os.path.join(root, "transformer"))
    tkw = dict(
        torch_dtype=dtype,
        render_encoder_kwargs={},
        use_embodiment_adapter=use_embodiment_adapter,
        embodiment_kwargs=embodiment_kwargs,
        legacy_render_variant=legacy_render_variant,
        v2_adapter_kwargs=v2_adapter_kwargs,
        v3_adapter_kwargs=v3_adapter_kwargs,
    )
    if local_transformer:
        dit = WanTransformerRenderConditioned.from_pretrained(
            os.path.join(root, "transformer"), **tkw
        )
    else:
        dit = WanTransformerRenderConditioned.from_pretrained(
            root, subfolder="transformer", **tkw
        )
    return dit


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if getattr(args, "resume_render_ckpt", None):
        rp = str(args.resume_render_ckpt).strip()
        if not rp:
            args.resume_render_ckpt = None
        elif int(getattr(args, "epoch_offset", 0)) <= 0:
            raise SystemExit(
                "[FSDP train] --resume_render_ckpt requires --epoch_offset > 0 "
                "(e.g. 30 after finishing epoch_29) so new epoch_* directories "
                "do not overwrite checkpoints from the earlier run."
            )

    # We manage DiT storage dtype manually via `_cast_module_for_fsdp`, and
    # never want Accelerate's FSDP path to upcast bf16 params to fp32 master
    # weights -- that doubles the sharded weight cost (e.g. ~8 -> ~16 GB/rank
    # for the 16B Wan DiT). FSDP still runs all-gather / reduce / compute in
    # the model's native (bf16) dtype, so we don't lose any speed.
    # Default NCCL collective timeout is 30 min, which is shorter than the
    # rank-0-only `_precompute_embeddings` walk (~5 sec/video × N samples → 70+ min
    # for 881 videos). Bump to 4 h so ranks 1..N-1 wait patiently at the
    # post-precompute `broadcast_object_list` instead of being killed by the
    # NCCL watchdog. Subsequent runs hit the on-disk embed cache and never pay
    # this wall-clock again.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=4))],
    )
    # Snapshot the BASE output dir before _setup_run_dir_and_logging mutates
    # it to `<base>/run_<timestamp>/`. The embed cache lives at the base so
    # every run shares it (instant relaunch); checkpoints/logs still land in
    # the per-run subdir.
    args._embed_cache_base_dir = args.output_dir
    args = _setup_run_dir_and_logging(args, accelerator)

    # `args.mixed_precision` here controls the DiT *storage* dtype only.
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }[args.mixed_precision]

    # Wan VAE compresses the temporal axis by 4 with a 1-frame "head" (so the
    # latent has T_lat = (num_frames - 1) // 4 + 1 frames). The Wan I2V
    # condition mask reshape (`view(1, -1, vae_t, ...)`) ALSO requires that
    # ``num_frames`` satisfies (num_frames - 1) % 4 == 0; otherwise the view
    # raises a cryptic shape error. Fail fast with a clear message.
    if (args.num_frames - 1) % 4 != 0 or args.num_frames < 5:
        raise SystemExit(
            f"--num_frames must satisfy (num_frames - 1) % 4 == 0 and >= 5 "
            f"(Wan VAE temporal layout). Got num_frames={args.num_frames}. "
            f"Try one of: 5, 9, 13, 17, 21, 33, 49, 65, 81, ..."
        )

    if accelerator.is_main_process:
        accelerator.print(f"[FSDP train] world_size={accelerator.num_processes}  device={accelerator.device}")
        accelerator.print(
            f"[FSDP train] trainable={args.trainable}  storage_dtype={args.mixed_precision} "
            f"(no fp32 master upcast; FSDP compute/reduce in {dtype})"
        )
        if getattr(args, "drop_render_conditioning", False):
            accelerator.print(
                "[FSDP train] drop_render_conditioning=True  →  forward uses vanilla I2V (render_latents=None); "
                "condition_usage_sanity (render vs shuffled-render) is skipped."
            )

    # ---------------------------------------------------------------
    # Phase 1: build dataset + preflight, then PRECOMPUTE all per-sample
    # tensors with one frozen encoder on the GPU at a time. After this
    # phase the encoders are freed -- the training loop only ever has
    # the (sharded) DiT on the GPU. This removes ~6+ GB of CPU<->GPU
    # encoder ping-pong from every step, which was the cause of the
    # periodic GPU-utilisation dips.
    # ---------------------------------------------------------------
    dataset = RenderI2VMetadataDataset(
        base_path=args.dataset_base_path,
        metadata_csv=args.metadata_csv,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        repeat=1,  # repeats are applied at the index level below
        action_stats_path=args.action_stats_path,
        action_field=args.action_field,
    )
    if accelerator.is_main_process:
        try:
            dataset.assert_local_files_exist()
        except FileNotFoundError as e:
            raise SystemExit(str(e)) from e
    accelerator.wait_for_everyone()

    # Encoder dtype: prefer the storage dtype if it's a low-precision float;
    # otherwise default to bf16 to keep the precompute footprint bounded.
    encoder_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16

    # Sharded precompute: every rank encodes its slice on its own GPU, then
    # results are all-gathered.  Cuts wall time ~world_size× vs. running on
    # rank 0 only (the previous design was forced by an FSDP/from_pretrained
    # interaction that we now sidestep via `safe_loading=True`).  The on-disk
    # cache is still a single shared file keyed by the FULL dataset, so any
    # relaunch with the same data instantly hits it.
    from pathlib import Path as _Path
    if getattr(args, "precompute_cache_path", None):
        embed_cache_path = _Path(args.precompute_cache_path)
    else:
        embed_cache_path = _Path(args._embed_cache_base_dir) / "embed_cache.pt"
    embed_cache_path.parent.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        accelerator.print(
            f"[FSDP train] sharded precompute of {len(dataset.rows)} samples across "
            f"{accelerator.num_processes} GPUs (encoder_dtype={encoder_dtype}) ..."
        )
    embed_cache, z_dim, scheduler = _precompute_embeddings_sharded(
        accelerator=accelerator,
        model_path=args.model_path,
        dataset=dataset,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        max_seq_len=args.max_sequence_length,
        encoder_dtype=encoder_dtype,
        ignore_prompts=bool(args.ignore_prompts),
        cache_path=embed_cache_path,
    )
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(
            f"[FSDP train] precompute done: {len(embed_cache)} cached samples "
            f"(latents + prompt + image + condition + tracks)"
        )

    # ----------------------------------------------------------------
    # Phase 1b: precompute the in-training eval caches (held-out + train).
    # Cheap: just a handful of samples. Done ONCE at startup; reused on
    # every per-epoch eval call. We use the same _precompute_embeddings
    # helper as the main precompute, but here run rank-0-only (the eval
    # set is small enough that sharding adds overhead, and only rank 0
    # consumes the cache during the eval block inside summon_full_params).
    # ----------------------------------------------------------------
    eval_cache_heldout_flow = None
    eval_cache_heldout_video = None
    eval_cache_train_video = None
    # All ranks precompute the eval cache (small dataset). This lets every
    # rank run the eval forward through the FSDP-sharded DiT, keeping
    # collectives synchronized. Doing eval on rank 0 only with
    # summon_full_params deadlocks because ranks 1-3 hit the next
    # collective ahead of rank 0 and the NCCL watchdog kills the job.
    if args.eval_every_n_epochs > 0:
        # Derive held-out CSV if not explicitly given. Convention:
        # train_metadata_train*.csv → train_metadata_test.csv (sibling file).
        if args.eval_csv_heldout:
            eval_csv = args.eval_csv_heldout
        else:
            mc = args.metadata_csv
            # Best-effort: try sibling test split.
            base = os.path.basename(mc)
            sibling = os.path.join(os.path.dirname(mc), "train_metadata_test.csv")
            eval_csv = sibling if os.path.isfile(sibling) else None
        if not eval_csv or not os.path.isfile(eval_csv):
            accelerator.print(
                f"[FSDP train eval] WARN: eval_csv_heldout not found "
                f"({eval_csv}); per-epoch eval disabled."
            )
        else:
            from world_model.wan_flow.train import _precompute_embeddings
            n_flow = max(0, int(args.eval_n_heldout_flow))
            n_vid_hold = max(0, int(args.eval_n_heldout_video) if args.eval_video_at_save else 0)
            n_vid_train = max(0, int(args.eval_n_train_video) if args.eval_video_at_save else 0)
            n_heldout_total = max(n_flow, n_vid_hold)
            if n_heldout_total > 0:
                eval_ds_h = RenderI2VMetadataDataset(
                    base_path=args.dataset_base_path,
                    metadata_csv=eval_csv,
                    num_frames=args.num_frames,
                    height=args.height, width=args.width, repeat=1,
                    action_stats_path=getattr(args, "action_stats_path", None),
                    action_field=getattr(args, "action_field", "state"),
                )
                eval_ds_h.rows = eval_ds_h.rows[:n_heldout_total]
                accelerator.print(
                    f"[FSDP train eval] precomputing {len(eval_ds_h.rows)} held-out "
                    f"samples from {eval_csv} (flow={n_flow}, video={n_vid_hold})"
                )
                eval_cache_full, _, eval_scheduler_obj = _precompute_embeddings(
                    model_path=args.model_path, dataset=eval_ds_h,
                    device=accelerator.device, height=args.height, width=args.width,
                    num_frames=args.num_frames,
                    max_seq_len=args.max_sequence_length,
                    encoder_dtype=encoder_dtype,
                    ignore_prompts=args.ignore_prompts,
                    cache_path=None, safe_loading=True,
                )
                eval_cache_heldout_flow  = eval_cache_full[:n_flow]
                eval_cache_heldout_video = eval_cache_full[:n_vid_hold]
            if n_vid_train > 0:
                eval_ds_t = RenderI2VMetadataDataset(
                    base_path=args.dataset_base_path,
                    metadata_csv=args.metadata_csv,
                    num_frames=args.num_frames,
                    height=args.height, width=args.width, repeat=1,
                    action_stats_path=getattr(args, "action_stats_path", None),
                    action_field=getattr(args, "action_field", "state"),
                )
                eval_ds_t.rows = eval_ds_t.rows[:n_vid_train]
                accelerator.print(
                    f"[FSDP train eval] precomputing {len(eval_ds_t.rows)} train-subset "
                    f"samples for overfit-gap signal"
                )
                eval_cache_train_video, _, _ = _precompute_embeddings(
                    model_path=args.model_path, dataset=eval_ds_t,
                    device=accelerator.device, height=args.height, width=args.width,
                    num_frames=args.num_frames,
                    max_seq_len=args.max_sequence_length,
                    encoder_dtype=encoder_dtype,
                    ignore_prompts=args.ignore_prompts,
                    cache_path=None, safe_loading=True,
                )
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    # ---------------------------------------------------------------
    # Phase 2: load only the DiT on each rank, FSDP-wrap, train.
    # ---------------------------------------------------------------
    dit = _load_dit_only(
        args.model_path, dtype,
        use_embodiment_adapter=bool(args.use_embodiment_adapter),
        embodiment_kwargs=args.embodiment_kwargs,
        legacy_render_variant=getattr(args, "legacy_render_variant", "egowm"),
        v2_adapter_kwargs=getattr(args, "v2_adapter_kwargs", None),
        v3_adapter_kwargs=getattr(args, "v3_adapter_kwargs", None),
    )
    n_meta = _materialize_meta_submodules(dit)
    # ``_materialize_meta_submodules`` re-runs ``reset_parameters`` on every
    # meta submodule, which wipes the explicit zero-inits used to guarantee
    # identity-at-init for the gated paths (DecoupledAdaLNHead.up,
    # MultiheadAttention.out_proj, etc.). Re-apply them.
    if hasattr(dit, "reset_zero_gates"):
        dit.reset_zero_gates()
    if accelerator.is_main_process and n_meta > 0:
        accelerator.print(f"[FSDP train] materialized {n_meta} meta submodules on CPU")
    if accelerator.is_main_process:
        accelerator.print(
            f"[FSDP train] conditioning path: "
            f"{'embodiment_adapter' if args.use_embodiment_adapter else 'legacy_egowm_eq5'}"
        )
    # Keep DiT on CPU until FSDP wraps it. This avoids per-rank device mismatch
    # (e.g. params on cuda:0 while rank device_id is cuda:1/2/3).
    dit = dit.cpu()
    casted = _cast_module_for_fsdp(dit, dtype)
    if accelerator.is_main_process and (casted["params"] > 0 or casted["buffers"] > 0):
        accelerator.print(
            f"[FSDP train] cast DiT floating tensors to {dtype}: "
            f"{casted['params'] / 1e6:.2f}M params, {casted['buffers'] / 1e6:.2f}M buffers"
        )

    dit.train()
    # Gradient checkpointing must be enabled **after** FSDP wraps the model
    # (see huggingface/accelerate#2178). Enabling it here caused checkpoint.backward +
    # sharded weights to misbehave under FULL_SHARD (e.g. crash at accelerator.backward).

    use_tracks_loss = args.lambda_tracks > 0 and any("gt_xy" in c for c in embed_cache)
    if args.trainable == "render_only":
        for p_ in dit.parameters():
            p_.requires_grad = False
        if not args.drop_render_conditioning:
            if args.use_embodiment_adapter:
                for p_ in dit.embodiment.parameters():
                    p_.requires_grad = True
            elif getattr(dit, "legacy_render_variant", "egowm") in ("v2", "v3"):
                # v2/v3: ActionConditionedTembAdapter (cross-attn) or
                # ActionRenderConcatAdapter (concat) is the entire trainable
                # render-conditioner. Both expose .parameters() under
                # `action_adapter.*`. No render_encoder / render_fuse /
                # render_gate.
                for p_ in dit.action_adapter.parameters():
                    p_.requires_grad = True
            else:
                # Legacy egowm / v1: render_encoder is always trainable.
                # render_fuse and render_gate exist only in the "v1" variant.
                for p_ in dit.render_encoder.parameters():
                    p_.requires_grad = True
                if getattr(dit, "legacy_render_variant", "egowm") == "v1":
                    for p_ in dit.render_fuse.parameters():
                        p_.requires_grad = True
                    dit.render_gate.requires_grad = True
            if use_tracks_loss:
                for p_ in dit.tracks_head.parameters():
                    p_.requires_grad = True

        if getattr(args, "unfreeze_last_n_blocks", 0) > 0:
            n_blocks = args.unfreeze_last_n_blocks
            for block in dit.blocks[-n_blocks:]:
                for p_ in block.parameters():
                    p_.requires_grad = True
            for p_ in dit.norm_out.parameters():
                p_.requires_grad = True
            for p_ in dit.proj_out.parameters():
                p_.requires_grad = True
            for p_ in dit.condition_embedder.time_proj.parameters():
                p_.requires_grad = True
    else:
        for p_ in dit.parameters():
            p_.requires_grad = True

    if args.disable_render_gate:
        if args.use_embodiment_adapter:
            # Adapter path has no single scalar render_gate; the equivalent
            # disable-attenuation hack would be to set every gate to 1.0
            # (render_adaln_gate, state_adaln_gate, per-adapter gate). That
            # bypasses the identity-at-init contract intentionally and is
            # only meant for ablation. Apply it on every gate.
            with torch.no_grad():
                dit.embodiment.render_adaln_gate.data.fill_(1.0)
                dit.embodiment.state_adaln_gate.data.fill_(1.0)
                for adapter in dit.embodiment.adapters.values():
                    adapter.gate.data.fill_(1.0)
            dit.embodiment.render_adaln_gate.requires_grad = False
            dit.embodiment.state_adaln_gate.requires_grad = False
            for adapter in dit.embodiment.adapters.values():
                adapter.gate.requires_grad = False
        else:
            # Legacy: only the v1 variant has a render_gate.
            if getattr(dit, "legacy_render_variant", "egowm") == "v1":
                with torch.no_grad():
                    dit.render_gate.data.fill_(1.0)
                dit.render_gate.requires_grad = False
            elif accelerator.is_main_process:
                accelerator.print(
                    "[FSDP train] disable_render_gate=True is a NO-OP for the "
                    "EgoWM-style legacy path (no gate exists)."
                )

    if getattr(args, "resume_render_ckpt", None):
        _load_resume_render_ckpt(dit, str(args.resume_render_ckpt), accelerator)

    # All DiT params (frozen + trainable) are kept in the storage ``dtype``
    # (typically bf16). FSDP requires uniform dtype within each
    # FlatParameter handle / wrap unit; ``use_orig_params=True`` only splits
    # by dtype AFTER flattening, so a mixed-dtype root unit fails at
    # ``_validate_tensors_to_flatten``. With --max_grad_norm=1.0 (default)
    # AdamW's exp_avg_sq stays in a range that bf16 can represent fine,
    # and the NaN/Inf guard inside the training loop catches any
    # divergence regardless. Tested empirically with the render-only +
    # tracks-head subset: gradient norms stay <1, AdamW state remains
    # finite over 100s of steps.
    trainable_params = [p for p in dit.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in dit.parameters())
    n_th = sum(p.numel() for p in dit.tracks_head.parameters())
    if accelerator.is_main_process:
        if args.use_embodiment_adapter:
            n_emb = sum(p.numel() for p in dit.embodiment.parameters())
            n_adapters = sum(
                p.numel()
                for a in dit.embodiment.adapters.values()
                for p in a.parameters()
            )
            # render_adaln_gate / state_adaln_gate are per-channel (D,);
            # report mean abs as a single scalar summary.
            gate_str = (
                f"render_adaln_gate={float(dit.embodiment.render_adaln_gate.detach().float().abs().mean().item()):.4g} "
                f"state_adaln_gate={float(dit.embodiment.state_adaln_gate.detach().float().abs().mean().item()):.4g}"
            )
            accelerator.print(
                f"[FSDP train] DiT params: {n_total / 1e9:.2f}B total, "
                f"{n_trainable / 1e6:.1f}M trainable ({100 * n_trainable / n_total:.2f}%); "
                f"embodiment={n_emb / 1e6:.2f}M (adapters={n_adapters / 1e6:.2f}M, "
                f"{len(dit.embodiment.adapters)} blocks); tracks_head={n_th / 1e6:.2f}M  "
                f"lambda_tracks={args.lambda_tracks}  use_tracks_loss={use_tracks_loss}  "
                f"disable_render_gate={args.disable_render_gate}  {gate_str}"
            )
        else:
            variant = getattr(dit, "legacy_render_variant", "egowm")
            if variant in ("v2", "v3"):
                n_aa = sum(p.numel() for p in dit.action_adapter.parameters())
                aa_cls = type(dit.action_adapter).__name__
                accelerator.print(
                    f"[FSDP train] DiT params: {n_total / 1e9:.2f}B total, "
                    f"{n_trainable / 1e6:.1f}M trainable ({100 * n_trainable / n_total:.2f}%); "
                    f"action_adapter={n_aa / 1e6:.2f}M  ({aa_cls}, variant={variant!r}); "
                    f"tracks_head={n_th / 1e6:.2f}M  "
                    f"lambda_tracks={args.lambda_tracks}  use_tracks_loss={use_tracks_loss}"
                )
            else:
                n_re = sum(p.numel() for p in dit.render_encoder.parameters())
                extra = (
                    f"render_gate={float(dit.render_gate.detach().float().item()):.4g}"
                    if variant == "v1" else "no gate"
                )
                accelerator.print(
                    f"[FSDP train] DiT params: {n_total / 1e9:.2f}B total, "
                    f"{n_trainable / 1e6:.1f}M trainable ({100 * n_trainable / n_total:.2f}%); "
                    f"render_encoder={n_re / 1e6:.2f}M (legacy variant={variant!r}); "
                    f"tracks_head={n_th / 1e6:.2f}M  "
                    f"lambda_tracks={args.lambda_tracks}  use_tracks_loss={use_tracks_loss}  "
                    f"disable_render_gate={args.disable_render_gate}  {extra}"
                )

    # Zero-init params (gates + tracks_head's final layer) get a higher LR.
    # AdamW step magnitude ≈ LR, so a param starting at exactly 0 needs many
    # steps × LR to reach a useful magnitude. The right multiplier depends on
    # the architecture:
    #   - Embodiment adapter: many gates need to escape exactly-zero init →
    #     10× usually right (default).
    #   - EgoWM-style legacy: there are no gates, only tracks_head.mlp[-1] is
    #     zero-init. tracks_head is a regular MLP (no SNR cancellation issue);
    #     small/no boost is fine. Drop to 1-3× for this variant.
    #   - v1 legacy: scalar render_gate has SNR cancellation, but with init=0.1
    #     it's not exactly zero. Mild boost (3-5×) usually right.
    # Override via the JSON config field ``gate_lr_multiplier``.
    GATE_LR_MULT = float(getattr(args, "gate_lr_multiplier", 10.0))
    def _is_zero_init_param(n: str) -> bool:
        return (
            n.endswith(".gate")
            or n.endswith("_gate")
            or "tracks_head.mlp.4" in n
        )
    _gate_param_ids = {
        id(p) for n, p in dit.named_parameters()
        if p.requires_grad and _is_zero_init_param(n)
    }
    gate_params  = [p for p in trainable_params if id(p) in _gate_param_ids]
    other_params = [p for p in trainable_params if id(p) not in _gate_param_ids]
    if accelerator.is_main_process:
        accelerator.print(
            f"[FSDP train] LR groups: zero_init_params={len(gate_params)} "
            f"(lr={args.learning_rate * GATE_LR_MULT:.1e}, {GATE_LR_MULT:g}× boost), "
            f"other_params={len(other_params)} (lr={args.learning_rate:.1e})"
        )
    optimizer = torch.optim.AdamW(
        [
            {"params": gate_params,  "lr": args.learning_rate * GATE_LR_MULT},
            {"params": other_params, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )

    n_samples = len(embed_cache)
    world_size = int(accelerator.num_processes)
    micro_pe = fsdp_micro_steps_per_epoch(n_samples, args.dataset_repeat, world_size)
    opt_total = optimizer_step_count(
        micro_pe * int(args.num_epochs), int(args.gradient_accumulation_steps)
    )
    lr_sched = build_lr_scheduler(
        optimizer,
        str(args.lr_scheduler),
        int(args.lr_warmup_steps),
        float(args.lr_min_ratio),
        opt_total,
        micro_pe=int(micro_pe),
        grad_accum=int(args.gradient_accumulation_steps),
        epoch_decay_start=int(args.lr_epoch_decay_start),
        epoch_decay_end=int(args.lr_epoch_decay_end),
    )

    dit, optimizer = accelerator.prepare(dit, optimizer)

    _dit_inner = accelerator.unwrap_model(dit)
    if hasattr(_dit_inner, "enable_gradient_checkpointing"):
        _dit_inner.enable_gradient_checkpointing()
        if accelerator.is_main_process:
            accelerator.print("[FSDP train] gradient checkpointing enabled on unwrap_model(dit) (post-FSDP wrap)")

    if accelerator.is_main_process and lr_sched is not None:
        extra = ""
        if str(args.lr_scheduler).strip().lower() in (
            "warmup_epoch_linear",
            "epoch_linear",
            "cosine_epoch",
        ):
            spep = max(1, optimizer_step_count(int(micro_pe), int(args.gradient_accumulation_steps)))
            eed, s1s = int(args.lr_epoch_decay_end), int(args.lr_epoch_decay_end) * spep
            if opt_total < s1s:
                extra = (
                    f"  [warn] total_opt_steps={opt_total} < end-of-decay step {s1s}  "
                    f"— raise --num_epochs or dataset_repeat.  "
                )
            else:
                extra = ""
            extra += (
                f"epoch_decay=[{int(args.lr_epoch_decay_start)},{eed})  "
                f"opt_step_decay=[{int(args.lr_epoch_decay_start) * spep},{s1s})"
            )
        accelerator.print(
            f"[FSDP train] lr_scheduler={args.lr_scheduler}  warmup_steps={args.lr_warmup_steps}  "
            f"lr_min_ratio={args.lr_min_ratio}  total_opt_steps={opt_total}  micro_steps/epoch/rank={micro_pe}"
            f"{extra}"
        )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- wandb -----------------------------------------------------------
    use_wandb = (
        accelerator.is_main_process
        and args.wandb_mode != "disabled"
        and os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true", "yes")
    )
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                config=vars(args),
                dir=args.output_dir,
            )
            accelerator.print(f"[wandb] logging to project={args.wandb_project} "
                              f"run={wandb.run.name} mode={args.wandb_mode}")
        except Exception as e:
            accelerator.print(f"[wandb] disabled (init failed: {e})")
            use_wandb = False

    n_samples = len(embed_cache)
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    device = accelerator.device
    global_step = 0
    forward_debug_done = False
    t0 = time.time()
    # Per-step flow loss varies massively with the random timestep
    # (low-sigma input ≈ clean → easy regression; mid-sigma input is
    # half-corrupted → hardest), with the random clean-sample (10 videos
    # are not equally close to Wan's training distribution), and with the
    # fresh noise draw. A single-step value swings between ~0.3 and ~1.8
    # even when the model is healthy. We expose three smoothed signals so
    # the trend is visible:
    #   1. EMA over all steps  (`train/loss_flow_ema`)            — fast but noisy on tiny runs
    #   2. Per-timestep-quartile mean (`train/flow_by_t_qK`)      — controls for sigma variance
    #   3. Per-epoch mean    (`train/loss_flow_epoch_mean`)       — best long-run trend signal
    flow_ema: Optional[float] = None
    flow_ema_decay = 0.95
    # Last grad norm we computed on a sync (non-accumulation) step. Within
    # a step, `grad_norm` is initialized to NaN and only overwritten when
    # `accelerator.sync_gradients` fires (every `gradient_accumulation_steps`
    # micro-steps). So if we display `grad_norm` directly, half the values
    # are the sentinel NaN. Keep the most recent real one for display.
    last_valid_grad_norm: float = float("nan")
    n_t = len(scheduler.timesteps)
    bucket_edges = [n_t * k // 4 for k in range(5)]   # [0, n/4, n/2, 3n/4, n]
    bucket_sum = [0.0, 0.0, 0.0, 0.0]
    bucket_cnt = [0, 0, 0, 0]

    for epoch in range(args.num_epochs):
        abs_e = int(args.epoch_offset) + int(epoch)
        # Reset per-epoch flow loss accumulator. Within an epoch each rank
        # processes a fixed disjoint slice of samples, so averaging over
        # the full epoch absorbs sample-, timestep- and noise-variance and
        # gives the cleanest trend signal across epochs.
        epoch_flow_sum = 0.0
        epoch_flow_cnt = 0
        # Deterministic shuffle so every rank computes the SAME global index
        # ordering, then we slice rank-by-rank to ensure each rank sees a
        # disjoint subset of samples per "step bucket". This replaces the
        # previous DataLoader+DistributedSampler path.
        g = torch.Generator()
        g.manual_seed(args.seed + abs_e)
        full_indices = torch.randperm(n_samples, generator=g).tolist()
        if args.dataset_repeat > 1:
            full_indices = full_indices * args.dataset_repeat
        # Pad up to a multiple of world_size so all ranks see the same step count.
        pad = (-len(full_indices)) % world_size
        if pad > 0:
            full_indices = full_indices + full_indices[:pad]
        my_indices = full_indices[rank::world_size]

        bar = tqdm(my_indices, disable=not accelerator.is_main_process, desc=f"epoch {abs_e}")
        for step, sample_idx in enumerate(bar):
            with accelerator.accumulate(dit):
                cached = embed_cache[sample_idx]
                clean_latents = cached["clean_latents"].to(device=device, dtype=dtype, non_blocking=True)
                render_latents = cached["render_latents"].to(device=device, dtype=dtype, non_blocking=True)
                prompt_embeds = cached["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
                image_embeds = cached["image_embeds"].to(device=device, dtype=dtype, non_blocking=True)
                condition = cached["condition"].to(device=device, dtype=dtype, non_blocking=True)
                # Optional 8-d (or 7-d) per-frame action stream — fed into the
                # action-aware adapter when the embodiment module's
                # ``use_action_aware_adaln`` is on. Cached as a (T, D) tensor
                # by precompute; we add the batch dim here.
                actions: Optional[torch.Tensor] = None
                if "actions" in cached:
                    actions = cached["actions"].to(
                        device=device, dtype=dtype, non_blocking=True,
                    ).unsqueeze(0)                                   # (1, T, D)

                # Tracks supervision (optional).
                gt_xy: Optional[torch.Tensor] = None
                gt_vis: Optional[torch.Tensor] = None
                query_xy: Optional[torch.Tensor] = None
                if use_tracks_loss and "gt_xy" in cached:
                    gt_xy = cached["gt_xy"].to(device=device, dtype=torch.float32, non_blocking=True)
                    gt_vis = cached["gt_vis"].to(device=device, dtype=torch.float32, non_blocking=True)
                    gt_xy = gt_xy.unsqueeze(0)              # (1, T, N, 2)
                    gt_vis = gt_vis.unsqueeze(0)            # (1, T, N)
                    if args.max_query_points > 0 and gt_xy.shape[2] > args.max_query_points:
                        N_full = gt_xy.shape[2]
                        sel = torch.randperm(N_full, device=device)[: args.max_query_points]
                        gt_xy = gt_xy[:, :, sel]
                        gt_vis = gt_vis[:, :, sel]
                    ref = max(0, min(args.ref_frame, gt_xy.shape[1] - 1))
                    query_xy = gt_xy[:, ref]                # (1, N, 2)

                with torch.no_grad():
                    noise = torch.randn_like(clean_latents)
                    idx = torch.randint(0, len(scheduler.timesteps), (1,))
                    # Keep timesteps in scheduler-native float32. Wan's flow-matching
                    # scheduler emits non-integer values (e.g. 979.6122). Casting to
                    # bf16/fp16 quantizes them and `.long()` truncates them, which
                    # corrupts the sin/cos time embedding and noise scaling lookups.
                    timestep_scalar = scheduler.timesteps[idx].to(device=device).float()
                    timestep_batch = timestep_scalar.expand(clean_latents.shape[0])
                    noisy_latents = scheduler.scale_noise(clean_latents, timestep_batch, noise)
                    noisy_latents[:, :, 0:1] = clean_latents[:, :, 0:1]

                latent_model_input = torch.cat([noisy_latents, condition], dim=1)
                target = noise - clean_latents

                # Render-dropout for CFG: with probability ``render_dropout_prob``,
                # train this step *without* the render condition so the model
                # learns a useful unconditional branch. Decision must be the
                # same on every rank to keep FSDP synchronized; we draw from
                # a deterministic generator seeded by (epoch, global_step) so
                # all ranks get the same coin flip without an extra collective.
                drop_render_step = False
                if (
                    args.render_dropout_prob > 0
                    and not args.drop_render_conditioning
                ):
                    drop_g = torch.Generator()
                    drop_g.manual_seed(int(args.seed) * 1_000_003 + abs_e * 65521 + global_step)
                    drop_render_step = bool(
                        torch.rand((1,), generator=drop_g).item() < args.render_dropout_prob
                    )

                forward_kwargs = dict(
                    hidden_states=latent_model_input,
                    timestep=timestep_batch,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    render_latents=(
                        None
                        if (args.drop_render_conditioning or drop_render_step)
                        else render_latents
                    ),
                    return_dict=False,
                )
                # Mirror render-dropout on actions: when the render is dropped
                # this step (CFG branch), drop actions too so the
                # unconditional branch is truly action-free.
                if actions is not None and not (
                    args.drop_render_conditioning or drop_render_step
                ):
                    forward_kwargs["actions"] = actions
                if query_xy is not None and not drop_render_step:
                    # When the render is dropped this step, the tracks-head
                    # path is also disabled (it requires the conditioned
                    # forward) -- fall back to flow-only loss for this step.
                    forward_kwargs["query_xy"] = query_xy
                    forward_kwargs["track_T"] = int(gt_xy.shape[1])

                model_out = dit(**forward_kwargs)
                pred = model_out[0]
                pred_tracks = model_out[1] if (query_xy is not None and len(model_out) > 1) else None

                loss_flow = F.mse_loss(pred[:, :, 1:].float(), target[:, :, 1:].float())
                if (
                    args.forward_debug_dir
                    and (not forward_debug_done)
                    and accelerator.is_main_process
                    and int(epoch) == int(args.forward_debug_epoch)
                    and int(step) == int(args.forward_debug_step_in_epoch)
                ):
                    dbg = (
                        args.forward_debug_dir
                        if os.path.isabs(args.forward_debug_dir)
                        else os.path.join(args.output_dir, args.forward_debug_dir)
                    )
                    save_forward_debug_bundle(
                        dbg,
                        dataset=dataset,
                        sample_idx=int(sample_idx),
                        num_frames=int(args.num_frames),
                        height=int(args.height),
                        width=int(args.width),
                        pred=pred,
                        target=target,
                        noise=noise,
                        noisy_latents=noisy_latents,
                        clean_latents=clean_latents,
                        render_latents=render_latents,
                        condition=condition,
                        latent_model_input=latent_model_input,
                        image_embeds=image_embeds,
                        prompt_embeds=prompt_embeds,
                        timestep_value=float(timestep_scalar.detach().float().item()),
                        loss_flow=float(loss_flow.detach().item()),
                        rgb_from_disk=bool(args.forward_debug_rgb_from_disk),
                        rgb_from_disk_fps=int(args.forward_debug_fps),
                        model_path=str(args.model_path) if args.forward_debug_rgb_from_disk else None,
                    )
                    forward_debug_done = True
                if pred_tracks is not None:
                    diff = F.smooth_l1_loss(
                        pred_tracks.float(), gt_xy.float(), reduction="none"
                    ).sum(-1)
                    vis_sum = gt_vis.sum().clamp(min=1.0)
                    loss_tracks = (diff * gt_vis).sum() / vis_sum
                else:
                    loss_tracks = torch.zeros((), device=device, dtype=loss_flow.dtype)
                loss = loss_flow + args.lambda_tracks * loss_tracks

                # ---- NaN/Inf guard on the loss before backward ----
                # Reduce across ranks so we make a CONSISTENT skip decision
                # on every rank (an FSDP step where some ranks call backward
                # and others don't would deadlock at the all-reduce).
                loss_finite_local = torch.isfinite(loss.detach()).to(torch.float32)
                loss_finite_all = accelerator.gather_for_metrics(
                    loss_finite_local.reshape(1)
                )
                loss_is_finite = bool(loss_finite_all.min().item() > 0.5)

                grad_norm = torch.tensor(float("nan"), device=device)
                if not loss_is_finite:
                    if accelerator.is_main_process:
                        accelerator.print(
                            f"[NaN-guard] step {global_step}: non-finite loss "
                            f"(flow={float(loss_flow):.4g}, tracks={float(loss_tracks):.4g}). "
                            f"{'Aborting' if args.abort_on_nan else 'Skipping step.'}"
                        )
                    if args.abort_on_nan:
                        raise RuntimeError(
                            f"Non-finite loss at step {global_step}: "
                            f"flow={float(loss_flow):.4g}, tracks={float(loss_tracks):.4g}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                else:
                    accelerator.backward(loss)
                    # One-time post-backward diagnostic: confirm gradients are
                    # actually flowing into the trainable params. Reports per-
                    # group grad-norm contribution so we can see whether
                    # render_encoder, render_fuse, and tracks_head all receive
                    # signal. (render_fuse is zero-init so render_encoder gets
                    # zero grad on step 0; that's expected and recovers on step 1+.)
                    if global_step == 0 and accelerator.is_main_process:
                        inner = accelerator.unwrap_model(dit)
                        if args.use_embodiment_adapter:
                            groups = [("embodiment", inner.embodiment)]
                        elif getattr(inner, "legacy_render_variant", "egowm") in ("v2", "v3"):
                            # v2/v3: only the action_adapter (cross-attn or
                            # concat adapter); no render_encoder.
                            groups = [("action_adapter", inner.action_adapter)]
                        else:
                            # egowm: render_encoder only. v1 also adds render_fuse.
                            groups = [("render_encoder", inner.render_encoder)]
                            if getattr(inner, "legacy_render_variant", "egowm") == "v1":
                                groups.append(("render_fuse", inner.render_fuse))
                        if use_tracks_loss:
                            groups.append(("tracks_head", inner.tracks_head))
                        msg_parts = []
                        for name, mod in groups:
                            local_sq = torch.tensor(0.0, device=device)
                            n_grad = 0
                            n_tot = 0
                            for p in mod.parameters():
                                n_tot += 1
                                if p.grad is not None:
                                    n_grad += 1
                                    local_sq = local_sq + p.grad.detach().float().pow(2).sum()
                            msg_parts.append(
                                f"{name}: local_norm={float(local_sq.sqrt()):.4g} "
                                f"({n_grad}/{n_tot} params have .grad)"
                            )
                        accelerator.print("[grad-check] " + " | ".join(msg_parts))
                    # Compute global pre-clip grad norm. Under FSDP we MUST go
                    # through ``model.clip_grad_norm_`` (the FSDP-attached
                    # method) -- ``accelerator.clip_grad_norm_`` only delegates
                    # to FSDP when the parameters list equals
                    # ``model.parameters()`` exactly. We pass a SUBSET
                    # (render_only trainable params), so accelerator falls
                    # back to ``torch.nn.utils.clip_grad_norm_`` which only
                    # sees the local shard and returns 0 / per-rank-different
                    # values. ``dit.clip_grad_norm_`` walks all params (frozen
                    # ones have ``.grad is None`` and are skipped), so the
                    # result is the true global norm of the trainable grads.
                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.FSDP:
                            gn = dit.clip_grad_norm_(args.max_grad_norm)
                        else:
                            gn = accelerator.clip_grad_norm_(
                                trainable_params, max_norm=args.max_grad_norm
                            )
                        if gn is not None:
                            grad_norm = gn.detach().to(device)
                            if torch.isfinite(grad_norm).item():
                                last_valid_grad_norm = float(grad_norm)
                    # Rank-consistent grad-finite decision. Under FSDP,
                    # ``dit.clip_grad_norm_`` already returns a globally-reduced
                    # value so the per-rank isfinite check would agree -- but
                    # for DDP fallback or future config changes the local
                    # check can disagree across ranks, causing one rank to
                    # skip optimizer.step while the rest step → shard drift.
                    # Mirror the loss_finite all-reduce above for safety, and
                    # skip the check entirely on accumulation (non-sync) steps
                    # so the NaN sentinel (grad_norm starts as NaN) doesn't
                    # trip a spurious warning every accumulation micro-step.
                    if accelerator.sync_gradients:
                        grad_finite_local = torch.isfinite(grad_norm).reshape(1).to(torch.float32)
                        grad_finite_all = accelerator.gather_for_metrics(grad_finite_local)
                        grad_finite = bool(grad_finite_all.min().item() > 0.5)
                    else:
                        grad_finite = True
                    if not grad_finite:
                        if accelerator.is_main_process:
                            accelerator.print(
                                f"[NaN-guard] step {global_step}: non-finite grad_norm "
                                f"({float(grad_norm):.4g}). "
                                f"{'Aborting' if args.abort_on_nan else 'Skipping optimizer.step.'}"
                            )
                        if args.abort_on_nan:
                            raise RuntimeError(
                                f"Non-finite grad_norm at step {global_step}: {float(grad_norm):.4g}"
                            )
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        optimizer.step()
                        if lr_sched is not None:
                            lr_sched.step()
                        optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # Update the smoothed-loss EMA + per-bucket + per-epoch stats
            # EVERY step (not just on logging steps) so the trend signal
            # isn't subsampled.
            flow_g_each = accelerator.gather_for_metrics(loss_flow.detach().reshape(1)).mean()
            flow_val = float(flow_g_each)
            if flow_ema is None:
                flow_ema = flow_val
            else:
                flow_ema = flow_ema_decay * flow_ema + (1.0 - flow_ema_decay) * flow_val
            t_idx = int(idx.item())
            b = min(3, max(0, sum(1 for e in bucket_edges[1:] if t_idx >= e)))
            bucket_sum[b] += flow_val
            bucket_cnt[b] += 1
            epoch_flow_sum += flow_val
            epoch_flow_cnt += 1

            if global_step % args.log_every_n_steps == 0:
                # Average per-step losses across ranks so wandb / progress bar
                # reflect the global metric, not just rank 0's local batch.
                loss_g = accelerator.gather_for_metrics(loss.detach().reshape(1)).mean()
                flow_g = flow_g_each
                tracks_g = accelerator.gather_for_metrics(loss_tracks.detach().reshape(1)).mean()
                # FSDP shards 1-element scalar gates (render_adaln_gate,
                # state_adaln_gate, per-adapter gate, render_gate) across ranks,
                # leaving 0-element local shards on most ranks. Calling .item()
                # on those crashes with "Tensor with 0 elements cannot be
                # converted to Scalar". summon_full_params is a collective so
                # every rank enters; rank0_only gathers only onto rank 0.
                inner_dit = accelerator.unwrap_model(dit)
                # ``offload_to_cpu=True``: materialize the unsharded params on
                # CPU instead of GPU rank 0. We only read tiny scalar/per-channel
                # gate values from this — the GPU-side cost (~28 GB for Wan
                # 14B unsharded) was OOM'ing 44 GB cards (L40 / A6000).
                with FSDP.summon_full_params(
                    dit, writeback=False, recurse=True, rank0_only=True,
                    offload_to_cpu=True,
                ):
                    if accelerator.is_main_process:
                        elapsed = time.time() - t0
                        # Prefer the last sync-step grad norm; on accumulation
                        # steps `grad_norm` is the per-step NaN sentinel.
                        gn_disp = (
                            float(grad_norm)
                            if torch.isfinite(grad_norm).item()
                            else last_valid_grad_norm
                        )
                        sigma_frac = float(t_idx) / max(1, n_t - 1)
                        if args.use_embodiment_adapter:
                            if getattr(inner_dit.embodiment, "use_action_aware_adaln", False):
                                # action-aware gate is per-channel; show mean abs
                                gate_disp = float(
                                    inner_dit.embodiment.action_aware_adaln.gate.detach()
                                    .float().abs().mean().item()
                                )
                            else:
                                # per-channel (D,); mean abs as scalar summary
                                gate_disp = float(
                                    inner_dit.embodiment.render_adaln_gate.detach()
                                    .float().abs().mean().item()
                                )
                        else:
                            # Legacy: only the v1 variant has a render_gate.
                            if getattr(inner_dit, "legacy_render_variant", "egowm") == "v1":
                                gate_disp = float(inner_dit.render_gate.float().item())
                            else:
                                gate_disp = float("nan")    # no gate exists
                        bar.set_postfix(
                            loss=f"{float(loss_g):.3f}",
                            flow=f"{float(flow_g):.3f}",
                            ema=f"{flow_ema:.3f}",
                            sig=f"{sigma_frac:.2f}",
                            gate=f"{gate_disp:.4f}",
                            tr=f"{float(tracks_g):.3f}" if pred_tracks is not None else "off",
                            gn=f"{gn_disp:.3f}",
                            ok="Y" if loss_is_finite else "N",
                            step=global_step, t=f"{elapsed:.0f}s",
                        )
                        if use_wandb:
                            import wandb
                            log_payload = {
                                "train/loss": float(loss_g),
                                "train/loss_flow": float(flow_g),
                                "train/loss_flow_ema": float(flow_ema),
                                "train/loss_tracks": float(tracks_g),
                                "train/grad_norm": gn_disp,
                                "train/loss_is_finite": float(loss_is_finite),
                                "train/lambda_tracks": args.lambda_tracks,
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "train/timestep": float(timestep_scalar.item()),
                                "train/timestep_sigma_frac": sigma_frac,
                                "train/epoch": abs_e,
                                "train/elapsed_s": elapsed,
                                "train/drop_render_step": float(drop_render_step),
                            }
                            if args.use_embodiment_adapter:
                                # per-channel (D,) gates → log mean abs
                                log_payload["train/render_adaln_gate"] = float(
                                    inner_dit.embodiment.render_adaln_gate.detach()
                                    .float().abs().mean().item()
                                )
                                log_payload["train/state_adaln_gate"] = float(
                                    inner_dit.embodiment.state_adaln_gate.detach()
                                    .float().abs().mean().item()
                                )
                                for adapter_id, adapter in inner_dit.embodiment.adapters.items():
                                    # gate is now per-channel (D,); log mean abs
                                    # as the scalar summary. Same metric shape
                                    # as render_adaln_gate / state_adaln_gate.
                                    log_payload[f"train/adapter_{adapter_id}_gate"] = float(
                                        adapter.gate.detach().float().abs().mean().item()
                                    )
                                # Action-aware AdaLN: gate is per-channel (D,), so log
                                # aggregate stats. Bootstrap check: ``gate_norm`` should
                                # ramp from 0 to ~0.1-1.0 over the first epoch. If it
                                # stays at exactly 0, the FSDP _materialize_meta_submodules
                                # path re-introduced the dual-zero bug — investigate
                                # reset_zero_gates.
                                if getattr(inner_dit.embodiment, "use_action_aware_adaln", False):
                                    aa_gate = inner_dit.embodiment.action_aware_adaln.gate.detach().float()
                                    log_payload["train/action_aware_gate_mean"] = float(aa_gate.mean().item())
                                    log_payload["train/action_aware_gate_std"] = float(aa_gate.std().item())
                                    log_payload["train/action_aware_gate_norm"] = float(aa_gate.norm().item())
                                    log_payload["train/action_aware_gate_max_abs"] = float(aa_gate.abs().max().item())
                            else:
                                lrv_log = getattr(inner_dit, "legacy_render_variant", "egowm")
                                if lrv_log == "v1":
                                    log_payload["train/render_gate"] = float(
                                        inner_dit.render_gate.float().item()
                                    )
                                elif lrv_log in ("v2", "v3"):
                                    # v2/v3 share the same out_proj diagnostic:
                                    # zero-init weight norm should ramp from 0
                                    # as gradient flows. Stays-at-0 would
                                    # indicate broken init (e.g. an MHA-zero
                                    # trap in v2 or a meta-materialization bug
                                    # in v3).
                                    op = inner_dit.action_adapter.out_proj.weight.detach().float()
                                    log_payload["train/action_adapter_out_proj_norm"] = float(op.norm().item())
                                    log_payload["train/action_adapter_out_proj_max_abs"] = float(op.abs().max().item())
                            # Per-quartile (by timestep) running-average flow loss.
                            # The "mid" buckets (Q1, Q2) are the hardest and are
                            # the best signal of actual learning progress.
                            for k in range(4):
                                if bucket_cnt[k] > 0:
                                    log_payload[f"train/flow_by_t_q{k}"] = bucket_sum[k] / bucket_cnt[k]
                            wandb.log(log_payload, step=global_step)

        # End-of-epoch summary: mean flow loss over the epoch. With many
        # steps this averages out timestep/sample/noise variance; with only
        # a handful of steps (tiny dataset × many GPUs) the mean itself is
        # still very noisy — use ``ema`` and W&B ``train/flow_by_t_q*``.
        epoch_mean = epoch_flow_sum / max(1, epoch_flow_cnt)
        few_steps = epoch_flow_cnt < 32
        if accelerator.is_main_process:
            hint = (
                "  |  (epoch mean is noisy: few steps/epoch — watch ema & flow_by_t_q*)"
                if few_steps
                else ""
            )
            accelerator.print(
                f"[epoch {abs_e}] mean flow_loss = {epoch_mean:.4f} "
                f"over {epoch_flow_cnt} steps  |  ema = {flow_ema:.4f}{hint}"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss_flow_epoch_mean": epoch_mean,
                    "train/epoch_done": abs_e,
                }, step=global_step)

        should_save = (
            (epoch + 1) % args.save_every_n_epochs == 0
            or epoch == args.num_epochs - 1
        )
        if should_save:
            # FSDP.summon_full_params is a collective call: all ranks must enter.
            # rank0_only=True ensures only rank 0 actually populates the gathered weights.
            # offload_to_cpu=True keeps the full unsharded params on CPU, not GPU —
            # crucial for 44 GB cards (L40, A6000) where summoning on GPU OOMs.
            with FSDP.summon_full_params(
                dit, writeback=False, recurse=True, rank0_only=True,
                offload_to_cpu=True,
            ):
                if (
                    accelerator.is_main_process
                    and int(args.condition_usage_sanity_samples) > 0
                    and not getattr(args, "drop_render_conditioning", False)
                ):
                    stats = condition_usage_sanity(
                        accelerator.unwrap_model(dit),
                        embed_cache,
                        scheduler,
                        device,
                        dtype,
                        int(args.condition_usage_sanity_samples),
                        int(args.seed) + int(abs_e),
                    )
                    if stats is not None:
                        accelerator.print(
                            "[sanity] condition usage: "
                            f"right={stats['loss_right']:.4f}  wrong={stats['loss_wrong']:.4f}  "
                            f"gap={stats['loss_gap']:+.4f} ({100.0*stats['loss_gap_ratio']:+.1f}%)  "
                            f"wrong>right={100.0*stats['wrong_higher_frac']:.1f}% over {int(stats['samples'])} samples"
                        )
                        if use_wandb:
                            import wandb
                            wandb.log({
                                "sanity/cond_loss_right": stats["loss_right"],
                                "sanity/cond_loss_wrong": stats["loss_wrong"],
                                "sanity/cond_loss_gap": stats["loss_gap"],
                                "sanity/cond_loss_gap_ratio": stats["loss_gap_ratio"],
                                "sanity/cond_wrong_higher_frac": stats["wrong_higher_frac"],
                            }, step=global_step)

            accelerator.wait_for_everyone()
            save_dir = os.path.join(args.output_dir, f"epoch_{abs_e}")
            if accelerator.is_main_process:
                os.makedirs(save_dir, exist_ok=True)

            # ``accelerator.get_state_dict`` triggers an FSDP all-gather of the
            # full sharded model on rank 0. We always call it (it's a collective)
            # but in render_only mode we strip it down to just the small
            # render-conditioner + tracks-head subset on rank 0 before writing.
            full_state = accelerator.get_state_dict(dit)
            if args.trainable == "render_only":
                if accelerator.is_main_process:
                    if args.use_embodiment_adapter:
                        keep_prefixes = ["embodiment."]
                        keep_exact: set = set()
                    elif getattr(args, "legacy_render_variant", "egowm") in ("v2", "v3"):
                        # v2/v3: action_adapter (cross-attn or concat) is the
                        # entire trainable conditioner. No render_encoder /
                        # render_fuse / render_gate in either path.
                        keep_prefixes = ["action_adapter."]
                        keep_exact = set()
                    else:
                        keep_prefixes = ["render_encoder.", "render_fuse."]
                        # render_gate is a top-level Parameter (no submodule
                        # prefix). Match by exact key so it survives the strip.
                        keep_exact = {"render_gate"}
                    if use_tracks_loss:
                        keep_prefixes.append("tracks_head.")
                    if getattr(args, "unfreeze_last_n_blocks", 0) > 0:
                        n_blocks = args.unfreeze_last_n_blocks
                        for i in range(len(accelerator.unwrap_model(dit).blocks) - n_blocks, len(accelerator.unwrap_model(dit).blocks)):
                            keep_prefixes.append(f"blocks.{i}.")
                        keep_prefixes.append("norm_out.")
                        keep_prefixes.append("proj_out.")
                        keep_prefixes.append("condition_embedder.time_proj.")
                    keep_prefixes = tuple(keep_prefixes)
                    small_state = {
                        k: v.detach().to("cpu")
                        for k, v in full_state.items()
                        if k.startswith(keep_prefixes) or k in keep_exact
                    }
                    ckpt_path = os.path.join(save_dir, "render_conditioner.pt")
                    torch.save(small_state, ckpt_path)
                    n_saved = sum(v.numel() for v in small_state.values())
                    accelerator.print(
                        f"Saved render conditioner ({n_saved / 1e6:.2f}M params, "
                        f"{len(small_state)} tensors) to {ckpt_path}"
                    )
                # Free the gathered full state on every rank ASAP.
                del full_state
            else:
                unwrapped = accelerator.unwrap_model(dit)
                unwrapped.save_pretrained(
                    save_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=full_state,
                )
                del full_state
                if accelerator.is_main_process:
                    accelerator.print(f"Saved full transformer to {save_dir}")

        # ----------------------------------------------------------------
        # In-training eval: every eval_every_n_epochs we run flow-MSE on
        # held-out; at save epochs (if --eval_video_at_save), additionally
        # run a small video eval with PSNR/SSIM. All ranks enter the
        # summon_full_params collective; only rank 0 actually runs forward.
        # ----------------------------------------------------------------
        run_flow_eval = (
            args.eval_every_n_epochs > 0
            and eval_cache_heldout_flow is not None
            and ((epoch + 1) % args.eval_every_n_epochs == 0
                 or epoch == args.num_epochs - 1)
        )
        run_video_eval = (
            args.eval_video_at_save
            and should_save
            and (eval_cache_heldout_video is not None or eval_cache_train_video is not None)
        )
        if run_flow_eval:
            # Run flow-MSE eval on ALL ranks through the FSDP-sharded DiT.
            # The forward issues all_gather collectives layer by layer, which
            # requires every rank to participate. Running eval on rank 0 only
            # (via summon_full_params) deadlocks: ranks 1-3 race ahead to the
            # next training collective while rank 0 is mid-eval, NCCL watchdog
            # times out at 480 s and SIGABRTs the job.
            from world_model.wan_flow.eval_inline import flow_mse_eval
            dit.eval()
            t_e = time.time()
            m = flow_mse_eval(
                dit, eval_cache_heldout_flow, scheduler,
                device=device, dtype=dtype,
                num_timesteps_per_sample=int(args.eval_timesteps_per_sample),
                seed=args.seed + abs_e,
            )
            dit.train()
            if accelerator.is_main_process:
                accelerator.print(
                    f"[eval epoch {abs_e}] heldout flow_mse mean="
                    f"{m['eval/heldout_flow_mse_mean']:.4f}  "
                    f"({time.time() - t_e:.0f}s)"
                )
                if use_wandb:
                    import wandb
                    wandb.log(m, step=global_step)
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.print(f"Training complete. {global_step} steps in {time.time() - t0:.0f}s")
        if use_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
