
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
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from world_model.wan_flow.data import RenderI2VMetadataDataset
from world_model.wan_flow.model import (
    WanTransformerRenderConditioned,
)
from world_model.wan_flow.train import (
    _precompute_embeddings,
    _setup_run_dir_and_logging,
    build_lr_scheduler,
    condition_usage_sanity,
    fsdp_micro_steps_per_epoch,
    optimizer_step_count,
    save_forward_debug_bundle,
)


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
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()
        else:
            for p in m.parameters(recurse=False):
                with torch.no_grad():
                    p.zero_()
            for b in m.buffers(recurse=False):
                with torch.no_grad():
                    b.zero_()
    return len(meta_modules)


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
    p.add_argument("--output_dir", type=str, default="./checkpoints/wan_render_fsdp")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--dataset_repeat", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
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
    p.set_defaults(**{k: v for k, v in defaults.items() if k != "config"})
    args = p.parse_args(remaining)
    if not args.model_path or not args.dataset_base_path or not args.metadata_csv:
        p.error("Provide --model_path, --dataset_base_path, and --metadata_csv (via CLI or --config JSON).")
    return args


def _load_dit_only(model_path: str, dtype: torch.dtype) -> WanTransformerRenderConditioned:
    """
    Load *only* the render-conditioned DiT (no VAE / text / image encoders).
    Encoders are run once at startup by ``_precompute_embeddings`` and then
    freed; they are never on the GPU during the FSDP training loop.
    """
    root = model_path.rstrip("/")
    local_transformer = os.path.isdir(os.path.join(root, "transformer"))
    tkw = dict(torch_dtype=dtype, render_encoder_kwargs={})
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

    # We manage DiT storage dtype manually via `_cast_module_for_fsdp`, and
    # never want Accelerate's FSDP path to upcast bf16 params to fp32 master
    # weights -- that doubles the sharded weight cost (e.g. ~8 -> ~16 GB/rank
    # for the 16B Wan DiT). FSDP still runs all-gather / reduce / compute in
    # the model's native (bf16) dtype, so we don't lose any speed.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
    )
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

    # Run the precompute on RANK 0 ONLY, then broadcast the cache to the
    # other ranks. We can't run it on every rank because under FSDP /
    # ``fsdp_cpu_ram_efficient_loading=true`` the transformers
    # ``from_pretrained(..., device_map="cuda:N")`` path silently drops the
    # weights to CPU on non-main processes (rank 0 loads cuda:0 fine, but
    # ranks 1+ get CPU weights -- hence the ``cpu vs cuda:N`` device
    # mismatch in the embedding lookup). Loading on rank 0 only also halves
    # CPU RAM peak and avoids 4 simultaneous reads of the 10+ GB encoder
    # checkpoints.
    if accelerator.is_main_process:
        accelerator.print(
            f"[FSDP train] precomputing {len(dataset.rows)} samples on rank 0 "
            f"(text + image + VAE + tracks, encoder_dtype={encoder_dtype}) ..."
        )
        embed_cache, z_dim, scheduler = _precompute_embeddings(
            args.model_path, dataset, accelerator.device,
            args.height, args.width, args.num_frames, args.max_sequence_length,
            encoder_dtype=encoder_dtype,
        )
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()
    else:
        embed_cache, z_dim, scheduler = None, None, None

    # Broadcast precompute outputs from rank 0 to all other ranks. Cache
    # tensors are CPU-side and the per-rank cache total is small (~tens of
    # MB), so pickling + an NCCL broadcast of the byte buffer is cheap.
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and accelerator.num_processes > 1:
        # ``broadcast_object_list`` pickles on rank 0 to a tensor on the
        # caller-specified device, broadcasts via the active process group,
        # and unpickles on every rank. With NCCL the staging tensor MUST be
        # on the rank's GPU, so we pass ``accelerator.device``. The
        # individual cache items are CPU tensors that survive the round-
        # trip unchanged because pickle records their original device.
        obj_list = [embed_cache, z_dim, scheduler]
        dist.broadcast_object_list(obj_list, src=0, device=accelerator.device)
        embed_cache, z_dim, scheduler = obj_list[0], obj_list[1], obj_list[2]
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(
            f"[FSDP train] precompute done & broadcast: {len(embed_cache)} cached samples "
            f"(latents + prompt + image + condition + tracks)"
        )

    # ---------------------------------------------------------------
    # Phase 2: load only the DiT on each rank, FSDP-wrap, train.
    # ---------------------------------------------------------------
    dit = _load_dit_only(args.model_path, dtype)
    n_meta = _materialize_meta_submodules(dit)
    if accelerator.is_main_process and n_meta > 0:
        accelerator.print(f"[FSDP train] materialized {n_meta} meta submodules on CPU")
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
            for p_ in dit.render_encoder.parameters():
                p_.requires_grad = True
            for p_ in dit.render_fuse.parameters():
                p_.requires_grad = True
            # render_gate is a top-level nn.Parameter on dit (not inside a
            # submodule), so it isn't covered by render_encoder.parameters() or
            # render_fuse.parameters(); enable it explicitly.
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
        with torch.no_grad():
            dit.render_gate.data.fill_(1.0)
        dit.render_gate.requires_grad = False

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
        accelerator.print(
            f"[FSDP train] DiT params: {n_total / 1e9:.2f}B total, "
            f"{n_trainable / 1e6:.1f}M trainable ({100 * n_trainable / n_total:.2f}%); "
            f"tracks_head={n_th / 1e6:.2f}M  lambda_tracks={args.lambda_tracks}  "
            f"use_tracks_loss={use_tracks_loss}  disable_render_gate={args.disable_render_gate}  "
            f"render_gate={float(dit.render_gate.detach().float().item()):.4g}"
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

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
    n_t = len(scheduler.timesteps)
    bucket_edges = [n_t * k // 4 for k in range(5)]   # [0, n/4, n/2, 3n/4, n]
    bucket_sum = [0.0, 0.0, 0.0, 0.0]
    bucket_cnt = [0, 0, 0, 0]

    for epoch in range(args.num_epochs):
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
        g.manual_seed(args.seed + epoch)
        full_indices = torch.randperm(n_samples, generator=g).tolist()
        if args.dataset_repeat > 1:
            full_indices = full_indices * args.dataset_repeat
        # Pad up to a multiple of world_size so all ranks see the same step count.
        pad = (-len(full_indices)) % world_size
        if pad > 0:
            full_indices = full_indices + full_indices[:pad]
        my_indices = full_indices[rank::world_size]

        bar = tqdm(my_indices, disable=not accelerator.is_main_process, desc=f"epoch {epoch}")
        for step, sample_idx in enumerate(bar):
            with accelerator.accumulate(dit):
                cached = embed_cache[sample_idx]
                clean_latents = cached["clean_latents"].to(device=device, dtype=dtype, non_blocking=True)
                render_latents = cached["render_latents"].to(device=device, dtype=dtype, non_blocking=True)
                prompt_embeds = cached["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
                image_embeds = cached["image_embeds"].to(device=device, dtype=dtype, non_blocking=True)
                condition = cached["condition"].to(device=device, dtype=dtype, non_blocking=True)

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

                forward_kwargs = dict(
                    hidden_states=latent_model_input,
                    timestep=timestep_batch,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    render_latents=None if args.drop_render_conditioning else render_latents,
                    return_dict=False,
                )
                if query_xy is not None:
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
                        groups = [
                            ("render_encoder", inner.render_encoder),
                            ("render_fuse",    inner.render_fuse),
                        ]
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
                            grad_norm = gn.detach()
                    grad_finite = bool(torch.isfinite(grad_norm).item()) if torch.isfinite(grad_norm).numel() else True
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
                if accelerator.is_main_process:
                    elapsed = time.time() - t0
                    gn_disp = float(grad_norm) if torch.isfinite(grad_norm).item() else float("nan")
                    sigma_frac = float(t_idx) / max(1, n_t - 1)
                    bar.set_postfix(
                        loss=f"{float(loss_g):.3f}",
                        flow=f"{float(flow_g):.3f}",
                        ema=f"{flow_ema:.3f}",
                        sig=f"{sigma_frac:.2f}",
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
                            "train/epoch": epoch,
                            "train/elapsed_s": elapsed,
                        }
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
                f"[epoch {epoch}] mean flow_loss = {epoch_mean:.4f} "
                f"over {epoch_flow_cnt} steps  |  ema = {flow_ema:.4f}{hint}"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss_flow_epoch_mean": epoch_mean,
                    "train/epoch_done": epoch,
                }, step=global_step)

        should_save = (
            (epoch + 1) % args.save_every_n_epochs == 0
            or epoch == args.num_epochs - 1
        )
        if should_save:
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
                    int(args.seed) + int(epoch),
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
            save_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            if accelerator.is_main_process:
                os.makedirs(save_dir, exist_ok=True)

            # ``accelerator.get_state_dict`` triggers an FSDP all-gather of the
            # full sharded model on rank 0. We always call it (it's a collective)
            # but in render_only mode we strip it down to just the small
            # render-conditioner + tracks-head subset on rank 0 before writing.
            full_state = accelerator.get_state_dict(dit)
            if args.trainable == "render_only":
                if accelerator.is_main_process:
                    keep_prefixes = ["render_encoder.", "render_fuse."]
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
                    # render_gate is a top-level Parameter (no submodule
                    # prefix). Match by exact key so it survives the strip.
                    keep_exact = {"render_gate"}
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

    if accelerator.is_main_process:
        accelerator.print(f"Training complete. {global_step} steps in {time.time() - t0:.0f}s")
        if use_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
