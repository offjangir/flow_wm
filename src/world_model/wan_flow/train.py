"""
Flow-matching SFT for :class:`~world_model.wan_flow.model.WanTransformerRenderConditioned`.

At each timestep the DiT is conditioned on VAE-encoded render-video latents
(per-frame action conditioning) in addition to the usual Wan I2V first-frame
condition. Objective matches DiffSynth-style ``FlowMatchSFTLoss``
(target ``noise - clean_latents``).

Run (single-GPU smoke test / local dev — one process, no ``accelerate`` multi-launch)::

  cd wm && pip install -e .
  python -m world_model.wan_flow.train --config configs/train_drrobot.json

Multi-GPU (optional)::

  accelerate launch --num_processes <N> -m world_model.wan_flow.train --config ... --no-single_process
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    LambdaLR,
    LinearLR,
    LRScheduler,
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from world_model.wan_flow.data import RenderI2VMetadataDataset


def _normalize_latents(z: torch.Tensor, vae) -> torch.Tensor:
    mean = torch.tensor(vae.config.latents_mean, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    return (z - mean) * inv_std


def _dataset_index_length(n_samples: int, dataset_repeat: int) -> int:
    """Length of the per-epoch index list before world-size padding (matches train loops)."""
    if dataset_repeat and dataset_repeat > 1:
        return int(n_samples) * int(dataset_repeat)
    return int(n_samples)


def fsdp_micro_steps_per_epoch(n_samples: int, dataset_repeat: int, world_size: int) -> int:
    """Micro-steps each rank runs per epoch under the FSDP index padding scheme."""
    L = _dataset_index_length(n_samples, dataset_repeat)
    ws = max(1, int(world_size))
    Lp = L + ((-L) % ws)
    return Lp // ws


def single_gpu_micro_steps_per_epoch(n_samples: int, dataset_repeat: int) -> int:
    """Micro-steps per epoch in ``train.py`` (no world padding)."""
    return _dataset_index_length(n_samples, dataset_repeat)


def optimizer_step_count(micro_steps_total: int, gradient_accumulation_steps: int) -> int:
    """Number of ``optimizer.step()`` calls implied by micro-steps and grad accumulation."""
    ga = max(1, int(gradient_accumulation_steps))
    mt = max(0, int(micro_steps_total))
    return max(1, math.ceil(mt / ga))


@torch.no_grad()
def condition_usage_sanity(
    model: torch.nn.Module,
    embed_cache: list,
    scheduler,
    device: torch.device,
    dtype: torch.dtype,
    num_samples: int,
    seed: int,
) -> Optional[Dict[str, float]]:
    """
    Quick probe: compare flow loss with correct render latents vs intentionally
    mismatched render latents on a fixed subset. If wrong-render loss is not
    worse, the model is likely under-using the render condition.
    """
    n = len(embed_cache)
    m = max(0, min(int(num_samples), n))
    if m <= 0 or n < 2:
        return None

    g = torch.Generator()
    g.manual_seed(int(seed))
    ids = torch.randperm(n, generator=g)[:m].tolist()
    wrong_ids = ids[1:] + ids[:1]

    right_losses = []
    wrong_losses = []
    wrong_higher = 0
    for i, j in zip(ids, wrong_ids):
        c = embed_cache[i]
        c_wrong = embed_cache[j]
        clean = c["clean_latents"].to(device=device, dtype=dtype, non_blocking=True)
        cond = c["condition"].to(device=device, dtype=dtype, non_blocking=True)
        prompt = c["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        image = c["image_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        rr = c["render_latents"].to(device=device, dtype=dtype, non_blocking=True)
        rw = c_wrong["render_latents"].to(device=device, dtype=dtype, non_blocking=True)
        ar = c["actions"].to(device=device, dtype=dtype, non_blocking=True).unsqueeze(0) if "actions" in c else None
        aw = c_wrong["actions"].to(device=device, dtype=dtype, non_blocking=True).unsqueeze(0) if "actions" in c_wrong else None

        noise = torch.randn_like(clean)
        t_idx = torch.randint(0, len(scheduler.timesteps), (1,), generator=g)
        t_scalar = scheduler.timesteps[t_idx].to(device=device).float()
        t_batch = t_scalar.expand(clean.shape[0])
        noisy = scheduler.scale_noise(clean, t_batch, noise)
        noisy[:, :, 0:1] = clean[:, :, 0:1]

        latent_in = torch.cat([noisy, cond], dim=1)
        target = noise - clean

        # Use autocast to handle any mixed-precision modules (e.g. time_proj
        # in fp32). Note: do NOT call ``model.to(dtype=...)`` here -- that
        # mutates the model storage dtype mid-training, and this probe is
        # run between training epochs. The autocast wrapper is sufficient
        # for forward consistency.
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            kw_right = dict(
                hidden_states=latent_in, timestep=t_batch,
                encoder_hidden_states=prompt, encoder_hidden_states_image=image,
                render_latents=rr, return_dict=False,
            )
            if ar is not None:
                kw_right["actions"] = ar
            pred_right = model(**kw_right)[0]
            kw_wrong = dict(
                hidden_states=latent_in, timestep=t_batch,
                encoder_hidden_states=prompt, encoder_hidden_states_image=image,
                render_latents=rw, return_dict=False,
            )
            # Pair the wrong render with the wrong actions so the contrast
            # is across the *full* embodiment conditioning, not split.
            if aw is not None:
                kw_wrong["actions"] = aw
            pred_wrong = model(**kw_wrong)[0]
        l_right = float(F.mse_loss(pred_right[:, :, 1:].float(), target[:, :, 1:].float()))
        l_wrong = float(F.mse_loss(pred_wrong[:, :, 1:].float(), target[:, :, 1:].float()))
        right_losses.append(l_right)
        wrong_losses.append(l_wrong)
        wrong_higher += int(l_wrong > l_right)

    mean_right = float(sum(right_losses) / max(1, len(right_losses)))
    mean_wrong = float(sum(wrong_losses) / max(1, len(wrong_losses)))
    gap = mean_wrong - mean_right
    return {
        "samples": float(len(right_losses)),
        "loss_right": mean_right,
        "loss_wrong": mean_wrong,
        "loss_gap": gap,
        "loss_gap_ratio": float(gap / max(1e-8, mean_right)),
        "wrong_higher_frac": float(wrong_higher / max(1, len(right_losses))),
    }


def _mul_cosine_1_to_min_then_plateau(step: int, n_cosine: int, min_r: float) -> float:
    """
    Multiplicative LR factor in [1.0, min_r]: full LR at the first step of the segment,
    ``min_r`` at the end of a cosine, then **constant** ``min_r`` (no periodic restart).
    ``n_cosine`` = number of scheduler steps the cosine part spans (>=1).
    """
    s, n = int(step), int(n_cosine)
    if n <= 1 or min_r >= 0.999999:
        return float(min_r) if s >= 0 else 1.0
    if s >= n:
        return float(min_r)
    u = float(s) / float(max(1, n - 1))
    c = (1.0 + math.cos(math.pi * u)) / 2.0
    return min_r + (1.0 - min_r) * c


def _mul_warmup_then_cosine_plateau(
    step: int, warmup: int, total: int, min_r: float, *,
    warmup_1e6: bool = True,
) -> float:
    """
    After optional linear warmup, cosine from 1.0 to ``min_r`` over the remaining
    ``total - warmup`` steps, then constant ``min_r`` (avoids ``CosineAnnealingLR``'s
    post-``T_max`` restart).
    """
    s, w, tot = int(step), int(warmup), int(total)
    if w > 0 and s < w:
        if warmup_1e6:
            return 1e-6 + (1.0 - 1e-6) * (float(s) / float(max(1, w - 1)))
        return 1.0
    t = s - w
    cos_n = max(0, tot - w)
    if cos_n <= 0:
        return 1.0
    return _mul_cosine_1_to_min_then_plateau(t, cos_n, min_r)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: str,
    warmup_steps: int,
    min_lr_ratio: float,
    total_optimizer_steps: int,
    *,
    micro_pe: int = 1,
    grad_accum: int = 1,
    epoch_decay_start: int = 300,
    epoch_decay_end: int = 800,
) -> Optional[LRScheduler]:
    """Optional LR schedule for adapter / full-DiT training.

    ``cosine``: cosine from base LR to ``min_lr_ratio * base_lr`` over
    ``total_optimizer_steps`` optimizer steps, then **constant** at that floor (no restart).

    ``cosine_warmup``: optional linear warmup, then cosine + same plateau (no post-``T_max`` rise).

    ``cosine_epoch``: no warmup: hold, then **cosine** in ``[lr_epoch_decay_start, lr_epoch_decay_end)``,
    then **constant** at ``lr * lr_min_ratio`` (like ``epoch_linear`` but a cosine segment).

    ``epoch_linear`` / ``warmup_epoch_linear``: linear segment(s) in the same epoch window; then
    **constant** at the floor (already plateaued).

    ``constant``: returns ``None`` (caller keeps fixed ``learning_rate``).
    """
    mode = (lr_scheduler or "constant").strip().lower()
    if mode in ("constant", "none", ""):
        return None
    if mode in ("warmup_epoch_linear", "epoch_linear", "cosine_epoch"):
        use_cos = mode == "cosine_epoch"
        spe = max(1, optimizer_step_count(int(micro_pe), int(grad_accum)))
        es, ee = int(epoch_decay_start), int(epoch_decay_end)
        if es < 0 or ee <= es:
            raise ValueError(
                f"{mode}: need 0 <= lr_epoch_decay_start < lr_epoch_decay_end, got {es=} {ee=}"
            )
        s0, s1 = es * spe, ee * spe
        w = 0 if mode in ("epoch_linear", "cosine_epoch") else max(0, int(warmup_steps))
        min_r = float(min_lr_ratio)
        if not (0.0 <= min_r < 1.0):
            raise ValueError(
                f"{mode}: lr_min_ratio should be in [0, 1) (e.g. 0.01 for 1e-6 with 1e-4 base), got {min_r}"
            )
        if w > s0:
            raise ValueError(
                f"{mode}: lr_warmup_steps={w} exceeds decay start step {s0} "
                f"(= {es} * opt_steps/epoch {spe}). Reduce warmup, use epoch_linear / cosine_epoch, "
                f"or start decay later."
            )

        def lr_mul(step: int) -> float:
            s = int(step)
            if w > 0 and s < w:
                return 1e-6 + (1.0 - 1e-6) * (float(s) / float(max(1, w - 1)))
            if s < s0:
                return 1.0
            if s0 <= s < s1:
                span = s1 - s0
                if span <= 1:
                    return float(min_r)
                u = (float(s) - float(s0)) / float(span - 1)
                if use_cos:
                    c = (1.0 + math.cos(math.pi * u)) / 2.0
                    return min_r + (1.0 - min_r) * c
                p = min(1.0, u)
                return (1.0 - p) * 1.0 + p * min_r
            return min_r

        return LambdaLR(optimizer, lr_mul)

    if mode in ("cosine", "cosine_warmup"):
        total = max(1, int(total_optimizer_steps))
        w = 0
        if mode == "cosine_warmup":
            w = max(0, min(int(warmup_steps), max(0, total - 1)))
        min_r = float(min_lr_ratio)
        if not (0.0 <= min_r < 1.0):
            raise ValueError(
                f"{mode}: lr_min_ratio should be in [0, 1) (e.g. 0.01 for 1e-6 with 1e-4 base), got {min_r}"
            )
        if mode == "cosine_warmup" and w > 0 and total - w <= 0:
            return LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=max(1, w))

        return LambdaLR(
            optimizer,
            lambda s: _mul_warmup_then_cosine_plateau(int(s), w, total, min_r, warmup_1e6=True),
        )

    raise ValueError(
        f"Unknown lr_scheduler={lr_scheduler!r}; expected one of: constant, cosine, cosine_warmup, "
        f"cosine_epoch, epoch_linear, warmup_epoch_linear."
    )


def _materialize_meta_submodules(model: torch.nn.Module) -> None:
    """Replace meta tensors (from newly-added modules missing in the checkpoint) with
    freshly-initialized tensors on CPU using each module's default ``reset_parameters``.
    This avoids the ``Cannot copy out of meta tensor`` error when later calling ``.to(device)``.
    """
    meta_modules = {
        name: m
        for name, m in model.named_modules()
        if any(p.is_meta for p in m.parameters(recurse=False))
        or any(b.is_meta for b in m.buffers(recurse=False))
    }
    if not meta_modules:
        return
    for name, m in meta_modules.items():
        m.to_empty(device="cpu", recurse=False)
        # Note: PyTorch convention is mixed. Some modules expose the public
        # `reset_parameters` (e.g. nn.Linear, nn.LayerNorm, nn.Conv*); others
        # use the private `_reset_parameters` (most notably nn.MultiheadAttention).
        # If we only check the public name we silently fall through to the
        # zero-everything else-branch for MHA — which zero-inits in_proj_weight
        # and silently kills cross-attention output. Cost us a day of debugging.
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
    print(f"[train] materialized {len(meta_modules)} meta submodules: "
          + ", ".join(sorted(meta_modules)[:8])
          + (f", ... (+{len(meta_modules) - 8} more)" if len(meta_modules) > 8 else ""))


def _encode_video_normalized(vae, video_processor, frames, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    vid = video_processor.preprocess_video(frames, height=height, width=width)
    vid = vid.to(device=vae.device, dtype=dtype)
    z = vae.encode(vid).latent_dist.sample()
    return _normalize_latents(z, vae)


def save_forward_debug_bundle(
    out_dir: str,
    *,
    dataset: Any,
    sample_idx: int,
    num_frames: int,
    height: int,
    width: int,
    pred: torch.Tensor,
    target: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    clean_latents: torch.Tensor,
    render_latents: torch.Tensor,
    condition: torch.Tensor,
    latent_model_input: torch.Tensor,
    image_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timestep_value: float,
    loss_flow: float,
    rgb_from_disk: bool = False,
    rgb_from_disk_fps: int = 8,
    model_path: Optional[str] = None,
) -> None:
    """
    Save **exactly the tensors used in this step's forward and flow loss** (no second
    decode path that could differ from precompute).

    ``tensors.npz`` holds:

    - ``pred`` / ``target`` — full DiT output and flow target ``noise - clean_latents``.
    - ``pred_supervision`` / ``target_supervision`` — ``pred[:, :, 1:]`` and
      ``target[:, :, 1:]``, i.e. the pair ``loss_flow = mse(..., ...)`` is computed on.
    - ``noise``, ``noisy_latents``, ``clean_latents``, ``render_latents`` — same tensors
      as in the training step (GT latent vs render conditioning latent).
    - ``latent_model_input`` — ``cat([noisy_latents, condition], dim=1)`` passed as
      ``hidden_states`` to the DiT.
    - ``condition``, ``image_embeds``, ``prompt_embeds`` — I2V latent mask+first-frame
      stack, CLIP image tokens, and text embeddings for this step.

    CSV paths are written to ``meta.json`` for traceability only.

    If ``rgb_from_disk`` is True, also writes optional MP4/PNG from **re-reading** disk
    (may not match ``VideoProcessor`` + VAE precompute pixel-perfectly).
    """
    os.makedirs(out_dir, exist_ok=True)
    row = dataset.rows[int(sample_idx)]
    base = dataset.base_path

    def _abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(base, p)

    vpath = _abs(str(row["video"]))
    rpath = _abs(str(row["render"]))

    pred_s = pred[:, :, 1:]
    target_s = target[:, :, 1:]
    np.savez_compressed(
        os.path.join(out_dir, "tensors.npz"),
        pred=pred.detach().float().cpu().numpy(),
        target=target.detach().float().cpu().numpy(),
        pred_supervision=pred_s.detach().float().cpu().numpy(),
        target_supervision=target_s.detach().float().cpu().numpy(),
        noise=noise.detach().float().cpu().numpy(),
        noisy_latents=noisy_latents.detach().float().cpu().numpy(),
        clean_latents=clean_latents.detach().float().cpu().numpy(),
        render_latents=render_latents.detach().float().cpu().numpy(),
        latent_model_input=latent_model_input.detach().float().cpu().numpy(),
        condition=condition.detach().float().cpu().numpy(),
        image_embeds=image_embeds.detach().float().cpu().numpy(),
        prompt_embeds=prompt_embeds.detach().float().cpu().numpy(),
    )

    files: Dict[str, Optional[str]] = {"tensors_npz": "tensors.npz"}
    if rgb_from_disk:
        import imageio.v2 as imageio
        from PIL import Image
        from world_model.wan_flow.data import _load_video_frames

        def _pil_list_to_uint8_thwc(frames: List[Any]) -> np.ndarray:
            out = []
            for fr in frames:
                out.append(np.asarray(fr.resize((width, height), Image.BILINEAR), dtype=np.uint8))
            return np.stack(out, axis=0)

        gt_pils = _load_video_frames(vpath, num_frames)
        render_pils = _load_video_frames(rpath, num_frames)
        gt_u8 = _pil_list_to_uint8_thwc(gt_pils)
        rd_u8 = _pil_list_to_uint8_thwc(render_pils)
        gt_pils[0].resize((width, height), Image.BILINEAR).save(
            os.path.join(out_dir, "optional_disk_first_frame_resized.png")
        )
        for path, arr in (
            (os.path.join(out_dir, "optional_disk_gt_rgb.mp4"), gt_u8),
            (os.path.join(out_dir, "optional_disk_render_rgb.mp4"), rd_u8),
        ):
            w = imageio.get_writer(
                path, fps=rgb_from_disk_fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None
            )
            try:
                for f in arr:
                    w.append_data(f)
            finally:
                w.close()
        files["optional_disk_first_frame_resized"] = "optional_disk_first_frame_resized.png"
        files["optional_disk_gt_rgb"] = "optional_disk_gt_rgb.mp4"
        files["optional_disk_render_rgb"] = "optional_disk_render_rgb.mp4"
        if model_path:
            clip_png = os.path.join(out_dir, "optional_disk_first_frame_clip_vis.png")
            try:
                from transformers import AutoImageProcessor

                proc = AutoImageProcessor.from_pretrained(model_path.rstrip("/"), subfolder="image_processor")
                pv = proc(images=gt_pils[0], return_tensors="pt")["pixel_values"][0]
                t = pv.detach().float().cpu()
                lo, hi = float(t.min()), float(t.max())
                if hi <= 1.5:
                    vis = (t * 255.0).clamp(0, 255).byte().numpy()
                else:
                    vis = ((t - lo) / max(hi - lo, 1e-8) * 255.0).clamp(0, 255).byte().numpy()
                if vis.ndim == 3 and vis.shape[0] in (1, 3):
                    if vis.shape[0] == 1:
                        vis = np.repeat(vis, 3, axis=0)
                    vis = np.transpose(vis, (1, 2, 0))
                Image.fromarray(vis).save(clip_png)
                files["optional_disk_first_frame_clip_vis"] = "optional_disk_first_frame_clip_vis.png"
            except Exception as e:
                with open(os.path.join(out_dir, "optional_disk_clip_note.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{e}\n")

    meta = {
        "sample_idx": int(sample_idx),
        "video_path": vpath,
        "render_path": rpath,
        "num_frames": int(num_frames),
        "height": int(height),
        "width": int(width),
        "timestep": float(timestep_value),
        "loss_flow": float(loss_flow),
        "loss_flow_definition": "mse(pred[:, :, 1:], target[:, :, 1:].float()) where target = noise - clean_latents",
        "tensor_shapes": {
            "pred": list(pred.shape),
            "target": list(target.shape),
            "pred_supervision": list(pred_s.shape),
            "target_supervision": list(target_s.shape),
            "noise": list(noise.shape),
            "noisy_latents": list(noisy_latents.shape),
            "clean_latents": list(clean_latents.shape),
            "render_latents": list(render_latents.shape),
            "latent_model_input": list(latent_model_input.shape),
            "condition": list(condition.shape),
            "image_embeds": list(image_embeds.shape),
            "prompt_embeds": list(prompt_embeds.shape),
        },
        "files": files,
        "rgb_from_disk": bool(rgb_from_disk),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[forward_debug] wrote bundle to {out_dir}")


def _setup_run_dir_and_logging(args, accelerator):
    """Per-run output layout, used by both train.py and train_fsdp.py::

        args.output_dir / run_<YYYYMMDD_HHMMSS> /
            train.log     # rank-0 stdout+stderr, line-buffered
            args.json     # frozen CLI args for reproducibility
            epoch_<N>/    # checkpoint subdirs (saved by main loop)
            wandb/        # wandb local run dir (if enabled)

    Mutates ``args.output_dir`` in place so downstream code (checkpoint save,
    wandb ``dir=...``) automatically lands in the run-specific subdir. Also
    auto-fills ``args.wandb_run_name`` to the local run id if unset, for
    aligned local↔remote naming.

    The log tee runs on rank 0 only -- under FSDP all 4 ranks would otherwise
    write the same lines, producing 4x duplicated logs.
    """
    import datetime
    import json as _json
    import sys

    run_dir = None
    if accelerator.is_main_process:
        run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "train.log")

        class _Tee:
            """Duplicate writes to multiple streams; delegate isatty/fileno/etc.
            to the first (live terminal) stream so tqdm still animates."""
            def __init__(self, *streams):
                self.streams = streams
            def write(self, s):
                for st in self.streams:
                    try:
                        st.write(s)
                        st.flush()
                    except Exception:
                        pass
                return len(s) if isinstance(s, str) else 0
            def flush(self):
                for st in self.streams:
                    try:
                        st.flush()
                    except Exception:
                        pass
            def __getattr__(self, name):
                return getattr(self.streams[0], name)

        log_f = open(log_path, "a", buffering=1, encoding="utf-8")
        log_f.write(f"\n===== run start: {run_id} =====\n")
        sys.stdout = _Tee(sys.__stdout__, log_f)
        sys.stderr = _Tee(sys.__stderr__, log_f)

        with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
            _json.dump(vars(args), f, indent=2, default=str)

        if not getattr(args, "wandb_run_name", None):
            args.wandb_run_name = run_id

    # Broadcast the run dir + wandb run name from rank 0 so every rank writes
    # checkpoints to the same place. Use ``broadcast_object_list`` so the
    # values pickle/unpickle cleanly even if torch.distributed isn't up yet
    # (single-process accelerator: ``is_initialized`` is False, we skip).
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and accelerator.num_processes > 1:
            obj = [run_dir, getattr(args, "wandb_run_name", None)]
            dist.broadcast_object_list(obj, src=0, device=accelerator.device)
            run_dir = obj[0]
            args.wandb_run_name = obj[1]
    except Exception:
        pass

    if run_dir:
        args.output_dir = run_dir

    if accelerator.is_main_process:
        accelerator.print(f"[run] output_dir = {args.output_dir}")
        accelerator.print(f"[run] log_file   = {os.path.join(args.output_dir, 'train.log')}")
    return args


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre.parse_known_args()
    defaults: Dict[str, Any] = {}
    if pre_args.config:
        with open(pre_args.config, encoding="utf-8") as f:
            defaults = json.load(f)

    p = argparse.ArgumentParser(
        description="Train render-conditioned Wan 2.1 I2V (Diffusers).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="JSON file with default hyperparameters.")
    p.add_argument("--model_path", type=str, required=False, help="HF model id or local Diffusers Wan I2V root.")
    p.add_argument("--dataset_base_path", type=str, required=False)
    p.add_argument("--metadata_csv", type=str, required=False)
    p.add_argument(
        "--action_stats_path", type=str, default=None,
        help="JSON with per-dim mean/std for z-score normalizing the action "
             "stream (data_wan_1k/action_stats.json). When set, dataset "
             "samples include normalized 'actions' (num_frames, D).")
    p.add_argument(
        "--action_field", type=str, default="state",
        help="Which field in actions/<scene>.npz to use ('state' = 8-d "
             "joint+gripper; 'action' = 7-d cmd-target+gripper).")
    p.add_argument("--output_dir", type=str, default="./checkpoints/wan_render")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--dataset_repeat", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument(
        "--gate_lr_multiplier", type=float, default=10.0,
        help="LR multiplier for zero-init params. See train_fsdp.py for details.",
    )
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_warmup",
        choices=("constant", "cosine", "cosine_warmup", "cosine_epoch", "epoch_linear", "warmup_epoch_linear"),
        help="constant: fixed; cosine: cosine to lr*min over full run, then hold; "
        "cosine_warmup: warmup + that; cosine_epoch: then cosine in lr_epoch_* window, hold floor; "
        "epoch_linear: linear in that window; warmup_epoch_linear: warmup + that linear window.",
    )
    p.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=300,
        help="Optimizer steps of linear 1e-6→1× LR ramp; 0 = no warmup. "
        "Ignored for constant, cosine, epoch_linear, and cosine_epoch.",
    )
    p.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.1,
        help="For cosine: eta_min = lr * this. For warmup_epoch_linear: final LR = lr * this "
        "(e.g. 1e-5 = 1e-4 * 0.01 to reach 1e-6 from 1e-4).",
    )
    p.add_argument(
        "--lr_epoch_decay_start",
        type=int,
        default=300,
        help="(cosine_epoch / epoch_linear / warmup_epoch_linear) First epoch (0-based) where decay/ramp starts.",
    )
    p.add_argument(
        "--lr_epoch_decay_end",
        type=int,
        default=800,
        help="(cosine_epoch / epoch_linear / warmup_epoch_linear) LR reaches floor at first step of this epoch, then constant.",
    )
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--trainable",
        type=str,
        default="render_only",
        choices=("render_only", "full_dit"),
        help="render_only: render_encoder + render_fuse + tracks_head; "
             "full_dit: entire DiT (high VRAM). The tracks_head is always "
             "trained when --lambda_tracks > 0.",
    )
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=("no", "fp16", "bf16"))
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--single_process",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, require exactly one Accelerate process. Use `python -m world_model.wan_flow.train ...` "
        "for single-GPU tests; use `--no-single_process` with multi-GPU `accelerate launch`.",
    )
    # ---- auxiliary tracks-head supervision (optional) ----
    p.add_argument("--lambda_tracks", type=float, default=0.1,
                   help="Weight of the auxiliary smooth-L1 tracks loss "
                        "(set 0 to disable the head entirely).")
    p.add_argument("--ref_frame", type=int, default=0,
                   help="Index (in num_frames) of the reference frame whose "
                        "track positions are used as query points.")
    p.add_argument("--max_query_points", type=int, default=-1,
                   help="If > 0, randomly subsample this many query points "
                        "per training step (memory). -1 means use all available.")
    # ---- wandb logging ----
    p.add_argument("--wandb_project", type=str, default="wan_flow_drrobot")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=("online", "offline", "disabled"),
                   help="Use 'disabled' or set WANDB_DISABLED=true to skip wandb.")
    p.add_argument("--log_every_n_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Gradient clipping norm; also used as the NaN/Inf guard threshold")
    p.add_argument("--abort_on_nan", action="store_true",
                   help="If set, raise on first non-finite loss/grad instead of skipping the step")
    p.add_argument(
        "--condition_usage_sanity_samples",
        type=int,
        default=0,
        help="If > 0, at save time compare flow loss on correct render vs shuffled render "
             "for this many cached samples (conditioning-usage sanity probe).",
    )
    p.add_argument(
        "--forward_debug_dir",
        type=str,
        default=None,
        help="If set, after the first matching training step save tensors.npz: exact pred/target "
             "(and pred_supervision vs target_supervision used in loss_flow), noise, noisy/clean/render "
             "latents, latent_model_input, condition, image_embeds, prompt_embeds. "
             "Relative paths are resolved under --output_dir (the run folder).",
    )
    p.add_argument(
        "--forward_debug_rgb_from_disk",
        action="store_true",
        help="With --forward_debug_dir, also write optional_disk_*.mp4/png by re-reading CSV paths "
             "(not the same tensors as precompute/VAE; for rough visual check only).",
    )
    p.add_argument(
        "--forward_debug_epoch",
        type=int,
        default=0,
        help="0-based epoch at which to run the forward-debug dump (only if --forward_debug_dir is set).",
    )
    p.add_argument(
        "--forward_debug_step_in_epoch",
        type=int,
        default=0,
        help="0-based step index within that epoch's shuffled index list (main rank / single GPU).",
    )
    p.add_argument(
        "--forward_debug_fps",
        type=int,
        default=8,
        help="FPS for gt_before_encode.mp4 and render_before_encode.mp4 in the forward-debug bundle.",
    )
    p.add_argument(
        "--disable_render_gate",
        action="store_true",
        help="If set, force render_gate=1.0 and freeze it (no attenuation, no gate learning).",
    )
    p.add_argument(
        "--ignore_prompts",
        action="store_true",
        help="If set, use empty string for all text prompts during precompute.",
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
        choices=("egowm", "v1", "v2"),
        help="Which legacy (non-embodiment-adapter) variant: 'egowm' "
             "(per-token spatial, no actions, no gate), 'v1' (BACKWARD-COMPAT "
             "pooled + fuse + scalar gate), or 'v2' (NEW, "
             "ActionConditionedTembAdapter, per-token cross-attn over actions, "
             "zero-init out_proj, no gates). See train_fsdp.py for details.",
    )
    p.add_argument(
        "--v2_adapter_kwargs",
        type=lambda s: json.loads(s) if isinstance(s, str) else s,
        default=None,
        help="JSON dict forwarded to ActionConditionedTembAdapter when "
             "--legacy_render_variant=v2. Required key: action_dim.",
    )
    if defaults:
        p.set_defaults(**{k: v for k, v in defaults.items() if k != "config"})
    args = p.parse_args(remaining)
    if not args.model_path or not args.dataset_base_path or not args.metadata_csv:
        p.error("Provide --model_path, --dataset_base_path, and --metadata_csv (via CLI or --config JSON).")
    return args


_PRECOMPUTE_CACHE_VERSION = 2


def _compute_precompute_key(
    *, model_path, rows, base_path, num_frames, height, width,
    max_seq_len, ignore_prompts, has_tracks, has_actions=False,
    action_field="state", action_stats_path="",
) -> str:
    """Stable hash of every input that affects ``_precompute_embeddings`` output.

    Includes the dataset's video / render / tracks paths AND their on-disk
    sizes + mtimes — re-encoding a render or swapping the metadata csv
    invalidates the cache automatically.
    """
    import hashlib, json, os
    h = hashlib.blake2b(digest_size=16)
    h.update(json.dumps({
        "version": _PRECOMPUTE_CACHE_VERSION,
        "model_path": str(model_path),
        "num_frames": int(num_frames),
        "height": int(height),
        "width": int(width),
        "max_seq_len": int(max_seq_len),
        "ignore_prompts": bool(ignore_prompts),
        "has_tracks": bool(has_tracks),
        "has_actions": bool(has_actions),
        "action_field": str(action_field) if has_actions else "",
        "n_rows": len(rows),
    }, sort_keys=True).encode())
    if has_actions and action_stats_path:
        # Stats file mtime/size folded in so re-normalizing (e.g. recomputing
        # mean/std on a different split) invalidates the cache.
        try:
            _ast = os.stat(action_stats_path)
            h.update(json.dumps([
                "action_stats", str(action_stats_path), _ast.st_size, _ast.st_mtime,
            ], default=str).encode())
        except OSError:
            h.update(b"action_stats:missing")

    def _stat(p_rel: str) -> tuple:
        if not p_rel:
            return (None, 0, 0.0)
        p = p_rel if os.path.isabs(p_rel) else os.path.join(base_path, p_rel)
        try:
            st = os.stat(p)
            return (p, st.st_size, st.st_mtime)
        except OSError:
            return (p, -1, -1.0)

    cols = ["video", "render", "tracks"]
    if has_actions:
        cols.append("actions")
    for r in rows:
        for col in cols:
            h.update(json.dumps([col, _stat(r.get(col, ""))], default=str).encode())
        h.update(("\n" + str(r.get("prompt", ""))).encode("utf-8", errors="replace"))
    return h.hexdigest()


@contextlib.contextmanager
def _disable_fsdp_cpu_ram_efficient_loading():
    """Suspend ``FSDP_CPU_RAM_EFFICIENT_LOADING`` for the duration of the block.

    When ``fsdp_cpu_ram_efficient_loading: true`` is set in the accelerate
    YAML, accelerate exports ``FSDP_CPU_RAM_EFFICIENT_LOADING=1``. transformers'
    ``from_pretrained`` honours that flag by skipping actual weight materialisation
    on every non-rank-0 process — leaving those ranks with random/uninitialised
    parameters. That's the desired behaviour later on the DiT (FSDP broadcasts
    sharded weights from rank 0 during ``accelerator.prepare``), but it silently
    breaks any standalone encoder load (UMT5, CLIP) the precompute does on every
    rank: ranks 1..N produce NaN embeddings.

    Use this context manager around encoder ``from_pretrained`` calls so every
    rank loads real weights, then restore the env var so FSDP wrap of the DiT
    still gets the RAM-efficient path.
    """
    keys = ("FSDP_CPU_RAM_EFFICIENT_LOADING", "ACCELERATE_FSDP_CPU_RAM_EFFICIENT_LOADING")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _precompute_embeddings(
    model_path, dataset, device, height, width, num_frames, max_seq_len,
    encoder_dtype: torch.dtype = torch.bfloat16,
    ignore_prompts: bool = False,
    cache_path=None,
    *,
    rows=None,
    safe_loading: bool = False,
    save_cache: bool = True,
):
    """
    Load each encoder **directly to GPU** one at a time via ``device_map``.
    CPU RAM never holds a full model (~2-3 GB peak), so 32 GB system RAM is fine.
    GPU holds at most one encoder at a time (text encoder ≈ 23 GB is the largest).

    Pre-computes per-sample:
      - ``clean_latents``: VAE latents of the real-camera target video
      - ``render_latents``: VAE latents of the DrRobot render video (conditioning)
      - ``prompt_embeds``: UMT5 text embeddings
      - ``image_embeds``: CLIP penultimate hidden states of the first target frame
      - ``condition``: Wan I2V (mask + first-frame-VAE) 20-channel condition tensor
      - ``gt_xy`` (optional): tracks (num_frames, N, 2) in [-1, 1] (if dataset has tracks)
      - ``gt_vis`` (optional): visibility (num_frames, N) in {0, 1}

    If ``cache_path`` is given and a cache file exists with a matching key
    (model + dataset + resolution + prompts), it is loaded from disk in seconds
    instead of re-running encoders. On a miss, the freshly-computed cache is
    saved to ``cache_path`` for the next launch.
    """
    import gc, os
    from pathlib import Path
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.video_processor import VideoProcessor
    from transformers import AutoImageProcessor, AutoTokenizer, CLIPVisionModel, UMT5EncoderModel
    from world_model.wan_flow.data import _load_tracks_npz, _load_video_frames
    from world_model.wan_flow.model import WanVAEChunkedEncode

    root = model_path.rstrip("/")
    # `rows` override: lets the FSDP sharded wrapper feed each rank its slice
    # without needing a full sub-dataset.  Falls back to dataset.rows when None.
    if rows is None:
        rows = dataset.rows
    n = len(rows)
    dev_str = str(device)

    # If ``root`` is a HuggingFace repo id (e.g. "Wan-AI/Wan2.1-I2V-..."), resolve
    # it to the local snapshot path. transformers' tokenizer init calls
    # ``model_info(repo_id)`` for its mistral-regex patch whenever the path is
    # NOT recognised as local — that API hit is anonymous-rate-limited (1000 req
    # / 5 min) and breaks instantly under 4 ranks × many encoder loads. Passing
    # a local path makes ``is_local=True`` so the patch fast-paths without the
    # round-trip. Files must already be in the HF cache.
    if not os.path.isdir(root):
        try:
            from huggingface_hub import snapshot_download
            root = snapshot_download(root, local_files_only=True)
            print(f"[precompute] resolved HF repo id → local snapshot: {root}")
        except Exception as exc:
            print(f"[precompute] WARN: could not resolve {root!r} to local snapshot ({exc}); "
                  f"falling back to repo id (may hit HF rate limit)")

    # ── Cache: short-circuit if a valid on-disk cache exists ───────────────
    # When `rows` was explicitly overridden (sharded mode), skip the cache
    # check — the caller computes the FULL-dataset cache key and handles
    # read/write at its level.  Otherwise the cache key would be wrong
    # (covers only this rank's slice).
    cache_key = None
    cache_path = Path(cache_path) if cache_path else None
    if cache_path is not None and rows is dataset.rows:
        cache_key = _compute_precompute_key(
            model_path=root, rows=rows, base_path=dataset.base_path,
            num_frames=num_frames, height=height, width=width,
            max_seq_len=max_seq_len, ignore_prompts=ignore_prompts,
            has_tracks=dataset.has_tracks,
            has_actions=getattr(dataset, "has_actions", False),
            action_field=getattr(dataset, "action_field", "state"),
            action_stats_path=getattr(dataset, "_action_stats_path", "") or "",
        )
        if cache_path.is_file():
            print(f"[precompute] checking on-disk cache: {cache_path}")
            try:
                blob = torch.load(cache_path, map_location="cpu", weights_only=False)
                if blob.get("key") == cache_key:
                    print(
                        f"[precompute] HIT  ({len(blob['cache'])} samples, "
                        f"{cache_path.stat().st_size / 1e9:.2f} GB)  → skipping encoders"
                    )
                    return blob["cache"], blob["z_dim"], blob["scheduler"]
                print(
                    f"[precompute] MISS — cache key changed "
                    f"(stored={blob.get('key','?')[:16]}…  expected={cache_key[:16]}…)"
                )
            except Exception as exc:
                print(f"[precompute] cache load failed ({exc}); recomputing")

    def _load_target(row):
        vpath = row["video"]
        if not os.path.isabs(vpath):
            vpath = os.path.join(dataset.base_path, vpath)
        return _load_video_frames(vpath, num_frames)

    def _load_render(row):
        rpath = row["render"]
        if not os.path.isabs(rpath):
            rpath = os.path.join(dataset.base_path, rpath)
        return _load_video_frames(rpath, num_frames)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")
    video_processor = VideoProcessor(vae_scale_factor=8)

    if ignore_prompts:
        print(f"[precompute 1/3] text encoder → GPU, {n}× empty prompts (ignore_prompts) ...")
    else:
        print(f"[precompute 1/3] text encoder → GPU, encoding {n} prompts ...")
    if safe_loading:
        # Plain CPU load + .to(device): avoids the from_pretrained(device_map=cuda:N)
        # path which silently drops weights to CPU on non-rank-0 ranks under
        # `fsdp_cpu_ram_efficient_loading=true`. Also suspend the env var so
        # transformers actually materialises weights on every rank (otherwise
        # ranks 1..N get uninitialised UMT5 → NaN embeddings).
        # Use T5TokenizerFast directly (the class declared in tokenizer_config.json)
        # so we avoid AutoTokenizer's AutoConfig probe — the Wan repo has no
        # top-level config.json, and that probe surfaces as a misleading
        # "outgoing traffic disabled" error when local_files_only is forced on.
        from transformers import T5TokenizerFast
        with _disable_fsdp_cpu_ram_efficient_loading():
            tokenizer = T5TokenizerFast.from_pretrained(root, subfolder="tokenizer")
            text_encoder = UMT5EncoderModel.from_pretrained(
                root, subfolder="text_encoder", torch_dtype=encoder_dtype,
            ).to(device).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(root, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            root, subfolder="text_encoder", torch_dtype=encoder_dtype,
            device_map=dev_str, low_cpu_mem_usage=True,
        ).eval()
    all_prompt_embeds = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            p_text = "" if ignore_prompts else str(row["prompt"])
            tok = tokenizer(
                [p_text],
                padding="max_length", max_length=max_seq_len,
                truncation=True, add_special_tokens=True,
                return_attention_mask=True, return_tensors="pt",
            )
            input_ids = tok.input_ids.to(device)
            mask = tok.attention_mask.to(device)
            # Wan I2V's canonical text encoding: pass attention_mask, then
            # truncate each sample to its non-padded length and zero-pad back
            # to ``max_seq_len`` so padding tokens contribute zero embedding.
            embeds = text_encoder(input_ids, attention_mask=mask).last_hidden_state
            seq_lens = mask.gt(0).sum(dim=1).long()
            embeds = [u[: int(v)] for u, v in zip(embeds, seq_lens)]
            embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in embeds],
                dim=0,
            )
            all_prompt_embeds.append(embeds.cpu())
    del text_encoder, tokenizer
    torch.cuda.empty_cache(); gc.collect()
    print("[precompute 1/3] done")

    print(f"[precompute 2/3] image encoder → GPU, encoding {n} first frames ...")
    if safe_loading:
        with _disable_fsdp_cpu_ram_efficient_loading():
            img_proc = AutoImageProcessor.from_pretrained(root, subfolder="image_processor")
            image_encoder = CLIPVisionModel.from_pretrained(
                root, subfolder="image_encoder", torch_dtype=encoder_dtype,
            ).to(device).eval()
    else:
        img_proc = AutoImageProcessor.from_pretrained(root, subfolder="image_processor")
        image_encoder = CLIPVisionModel.from_pretrained(
            root, subfolder="image_encoder", torch_dtype=encoder_dtype,
            device_map=dev_str, low_cpu_mem_usage=True,
        ).eval()
    all_image_embeds = []
    with torch.no_grad():
        for i, row in enumerate(tqdm(rows, desc="[precompute 2/3] image enc", unit="vid")):
            frames = _load_target(row)
            pv = img_proc(images=frames[0], return_tensors="pt").pixel_values
            pv = pv.to(device=device, dtype=encoder_dtype)
            ie = image_encoder(pv, output_hidden_states=True).hidden_states[-2]
            all_image_embeds.append(ie.cpu())
    del image_encoder, img_proc
    torch.cuda.empty_cache(); gc.collect()
    print("[precompute 2/3] done")

    print(f"[precompute 3/3] VAE → GPU, encoding {n} target + render videos ...")
    vae = WanVAEChunkedEncode.from_pretrained(
        root, subfolder="vae", torch_dtype=encoder_dtype,
    )
    vae = vae.to(device).eval()
    z_dim = vae.config.z_dim
    vae_t = 4  # Wan VAE temporal compression
    latent_h, latent_w = height // 8, width // 8

    all_latents, all_render_latents, all_conditions = [], [], []
    with torch.no_grad():
        for i, row in enumerate(tqdm(rows, desc="[precompute 3/3] VAE", unit="vid")):
            frames = _load_target(row)
            render_frames = _load_render(row)

            clean_latents = _encode_video_normalized(
                vae, video_processor, frames, height, width, dtype=encoder_dtype,
            ).cpu()
            all_latents.append(clean_latents)

            render_latents = _encode_video_normalized(
                vae, video_processor, render_frames, height, width, dtype=encoder_dtype,
            ).cpu()
            all_render_latents.append(render_latents)

            # Wan I2V first-frame condition (mask + first_frame VAE + zeros), 20ch.
            first = video_processor.preprocess(frames[0], height=height, width=width)
            first = first.unsqueeze(2).to(device=device, dtype=encoder_dtype)  # (1, 3, 1, H, W)
            video_condition = torch.cat(
                [first, first.new_zeros(1, first.shape[1], num_frames - 1, height, width)], dim=2
            )
            latent_condition = vae.encode(video_condition).latent_dist.mode()
            latent_condition = _normalize_latents(latent_condition, vae)  # (1, 16, T_lat, Hl, Wl)

            mask_lat_size = torch.ones(1, 1, num_frames, latent_h, latent_w)
            mask_lat_size[:, :, 1:] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1].repeat_interleave(vae_t, dim=2)
            mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
            mask_lat_size = mask_lat_size.view(1, -1, vae_t, latent_h, latent_w).transpose(1, 2)
            mask_lat_size = mask_lat_size.to(device=device, dtype=latent_condition.dtype)

            condition = torch.cat([mask_lat_size, latent_condition], dim=1)  # (1, 20, T_lat, Hl, Wl)
            all_conditions.append(condition.cpu())
            print(f"  [{i+1}/{n}] encoded")
    del vae
    torch.cuda.empty_cache(); gc.collect()
    print("[precompute 3/3] done")

    # Optional per-sample track supervision: load + temporally subsample once
    # so the train loop just indexes into pre-tensorized tracks.
    all_tracks: list = []
    if dataset.has_tracks:
        for row in rows:
            t = row.get("tracks", "") or ""
            if not t:
                all_tracks.append(None)
                continue
            tpath = t if os.path.isabs(t) else os.path.join(dataset.base_path, t)
            # Re-project tracks into the training canvas (height, width) so they
            # stay aligned with the resized target video. See _load_tracks_npz.
            trajs, visibs, _img = _load_tracks_npz(
                tpath, num_frames,
                target_height=height, target_width=width,
            )
            all_tracks.append({"gt_xy": trajs, "gt_vis": visibs})
        n_with = sum(1 for t in all_tracks if t is not None)
        print(f"[precompute tracks] {n_with}/{n} samples have tracks")
    else:
        all_tracks = [None] * n

    # Optional per-sample action conditioning: pre-load + temporally subsample
    # once so the train loop just indexes into pre-tensorized actions.
    all_actions: list = []
    if getattr(dataset, "has_actions", False):
        from world_model.wan_flow.data import _load_actions_npz
        a_field = getattr(dataset, "action_field", "state")
        a_stats = getattr(dataset, "action_stats", None)
        for row in rows:
            a = row.get("actions", "") or ""
            if not a:
                all_actions.append(None)
                continue
            apath = a if os.path.isabs(a) else os.path.join(dataset.base_path, a)
            arr = _load_actions_npz(
                apath, num_frames, stats=a_stats, field=a_field,
            )                                                  # (num_frames, D)
            all_actions.append(torch.from_numpy(arr))
        n_with = sum(1 for a in all_actions if a is not None)
        print(f"[precompute actions] {n_with}/{n} samples have actions")
    else:
        all_actions = [None] * n

    cache = []
    for i in range(n):
        sample = {
            "clean_latents": all_latents[i],
            "render_latents": all_render_latents[i],
            "prompt_embeds": all_prompt_embeds[i],
            "image_embeds": all_image_embeds[i],
            "condition": all_conditions[i],
        }
        if all_tracks[i] is not None:
            sample["gt_xy"] = all_tracks[i]["gt_xy"]
            sample["gt_vis"] = all_tracks[i]["gt_vis"]
        if all_actions[i] is not None:
            sample["actions"] = all_actions[i]
        cache.append(sample)

    # Skip the cache write when the caller is shard-mode (rows overridden) or
    # explicitly disabled it.  In sharded mode the FULL-dataset cache is
    # written by the gather wrapper, not here.
    if cache_path is not None and cache_key is not None and save_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            torch.save(
                {"key": cache_key, "cache": cache, "z_dim": z_dim, "scheduler": scheduler,
                 "version": _PRECOMPUTE_CACHE_VERSION},
                tmp,
            )
            os.replace(tmp, cache_path)
            print(
                f"[precompute] saved cache → {cache_path} "
                f"({cache_path.stat().st_size / 1e9:.2f} GB, key={cache_key[:16]}…)"
            )
        except Exception as exc:
            print(f"[precompute] WARN: failed to write cache to {cache_path}: {exc}")
            if tmp.exists():
                try: tmp.unlink()
                except OSError: pass

    return cache, z_dim, scheduler


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    args = _setup_run_dir_and_logging(args, accelerator)
    if args.single_process and accelerator.num_processes != 1:
        raise SystemExit(
            "single_process is enabled but Accelerate started multiple processes "
            f"({accelerator.num_processes}). For single-GPU testing from train.py, run exactly:\n"
            "  cd wm && python -m world_model.wan_flow.train --config configs/train_drrobot.json\n"
            "Omit `accelerate launch` unless you pass `--no-single_process` and intend multi-GPU."
        )
    device = accelerator.device
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

    # Wan patch size (1,2,2): rough DiT token count after patchify (spatial only).
    # Render-conditioned path expands time-embeddings to every token; very large
    # (480p × long clips) commonly OOMs on a single 48 GB card with the full 14B DiT.
    if args.single_process and torch.cuda.is_available():
        latent_t = (int(args.num_frames) - 1) // 4 + 1
        latent_h, latent_w = int(args.height) // 8, int(args.width) // 8
        p_h, p_w = 2, 2
        n_tok = latent_t * (latent_h // p_h) * (latent_w // p_w)
        if n_tok > 12_000:
            accelerator.print(
                f"[warn] single-GPU: ~{n_tok} DiT tokens ({args.height}x{args.width}, "
                f"num_frames={args.num_frames}). This often exceeds 48 GB VRAM with the "
                f"14B render-conditioned forward. Use configs/train_drrobot_single_gpu.json "
                f"(smaller H/W and frames), or accelerate + world_model.wan_flow.train_fsdp "
                f"for full 480p-style training."
            )

    dataset = RenderI2VMetadataDataset(
        base_path=args.dataset_base_path,
        metadata_csv=args.metadata_csv,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        repeat=1,
        action_stats_path=args.action_stats_path,
        action_field=args.action_field,
    )
    try:
        dataset.assert_local_files_exist()
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    # Encoders (text/image/VAE) only need to compute frozen features once; running
    # them in fp32 wastes memory + is slow. Use bf16 by default, but if the user
    # explicitly chose fp16/bf16 for training, match that for consistency.
    encoder_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16

    # --- Phase 1: pre-encode with one encoder at a time (never >~10GB on GPU) ---
    embed_cache, _z_dim, scheduler = _precompute_embeddings(
        args.model_path, dataset, device,
        args.height, args.width, args.num_frames, args.max_sequence_length,
        encoder_dtype=encoder_dtype,
    )
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase 2: load ONLY the DiT onto GPU for training ---
    accelerator.print("[train] loading DiT to GPU ...")
    from world_model.wan_flow.model import WanTransformerRenderConditioned
    root = args.model_path.rstrip("/")
    local_transformer = os.path.isdir(os.path.join(root, "transformer"))
    # NOTE: Do not pass device_map / low_cpu_mem_usage=True here. Those place new modules
    # (render_encoder, render_fuse) on the `meta` device since they are absent from the
    # Wan checkpoint; a subsequent `dispatch_model -> model.to(device)` then fails with
    # "Cannot copy out of meta tensor; no data!". Load on CPU, materialize + init the new
    # modules, then .to(device) manually.
    # Load DiT in the user-selected mixed-precision dtype (or bf16 fallback for fp32
    # to keep weight memory bounded on consumer GPUs; Accelerator will autocast).
    dit_load_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    tkw = dict(
        torch_dtype=dit_load_dtype,
        render_encoder_kwargs={},
        use_embodiment_adapter=bool(args.use_embodiment_adapter),
        legacy_render_variant=getattr(args, "legacy_render_variant", "egowm"),
        v2_adapter_kwargs=getattr(args, "v2_adapter_kwargs", None),
    )
    if local_transformer:
        dit = WanTransformerRenderConditioned.from_pretrained(os.path.join(root, "transformer"), **tkw)
    else:
        dit = WanTransformerRenderConditioned.from_pretrained(root, subfolder="transformer", **tkw)

    _materialize_meta_submodules(dit)
    # Re-apply identity-at-init zero-inits that ``reset_parameters`` wiped during
    # meta materialization (DecoupledAdaLNHead.up, MultiheadAttention.out_proj, ...).
    if hasattr(dit, "reset_zero_gates"):
        dit.reset_zero_gates()
    accelerator.print(
        f"[train] conditioning path: "
        f"{'embodiment_adapter' if args.use_embodiment_adapter else 'legacy_egowm_eq5'}"
    )
    dit = dit.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    dit.train()
    dit.enable_gradient_checkpointing()

    use_tracks_loss = args.lambda_tracks > 0 and any("gt_xy" in c for c in embed_cache)
    if args.trainable == "render_only":
        for p_ in dit.parameters():
            p_.requires_grad = False
        if args.use_embodiment_adapter:
            for p_ in dit.embodiment.parameters():
                p_.requires_grad = True
        elif getattr(dit, "legacy_render_variant", "egowm") == "v2":
            # v2: ActionConditionedTembAdapter is the entire trainable
            # render-conditioner; no render_encoder / render_fuse / render_gate.
            for p_ in dit.action_adapter.parameters():
                p_.requires_grad = True
        else:
            # Legacy egowm / v1: render_encoder always trainable.
            for p_ in dit.render_encoder.parameters():
                p_.requires_grad = True
            if getattr(dit, "legacy_render_variant", "egowm") == "v1":
                for p_ in dit.render_fuse.parameters():
                    p_.requires_grad = True
                dit.render_gate.requires_grad = True
        if use_tracks_loss:
            for p_ in dit.tracks_head.parameters():
                p_.requires_grad = True

        # Mirrors the same logic in train_fsdp.py: unfreeze the last N
        # transformer blocks plus the head (norm_out, proj_out) and the
        # condition_embedder.time_proj so the AdaLN modulation path is
        # trainable. Without this, ``unfreeze_last_n_blocks`` in the config
        # is silently ignored on the single-process path → on render-dropout
        # steps the only trainable params (render_encoder) are skipped,
        # leaving the loss with no grad_fn and crashing backward.
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
            if getattr(dit, "legacy_render_variant", "egowm") == "v1":
                with torch.no_grad():
                    dit.render_gate.data.fill_(1.0)
                dit.render_gate.requires_grad = False
            else:
                print("[train] disable_render_gate is a NO-OP for the EgoWM-style "
                      "legacy variant (no gate exists).")

    trainable = [p for p in dit.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in dit.parameters())
    n_th = sum(p.numel() for p in dit.tracks_head.parameters())
    if args.use_embodiment_adapter:
        n_emb = sum(p.numel() for p in dit.embodiment.parameters())
        # render_adaln_gate / state_adaln_gate are per-channel (D,);
        # report mean abs as a single scalar summary.
        gate_str = (
            f"render_adaln_gate={float(dit.embodiment.render_adaln_gate.detach().float().abs().mean().item()):.4g} "
            f"state_adaln_gate={float(dit.embodiment.state_adaln_gate.detach().float().abs().mean().item()):.4g}"
        )
        accelerator.print(
            f"[train] DiT: {n_total/1e9:.2f}B total, {n_trainable/1e6:.1f}M trainable "
            f"({100*n_trainable/n_total:.2f}%); embodiment={n_emb/1e6:.2f}M; "
            f"tracks_head: {n_th/1e6:.2f}M  lambda_tracks={args.lambda_tracks}  "
            f"use_tracks_loss={use_tracks_loss}  disable_render_gate={args.disable_render_gate}  {gate_str}"
        )
    else:
        variant = getattr(dit, "legacy_render_variant", "egowm")
        if variant == "v2":
            n_aa = sum(p.numel() for p in dit.action_adapter.parameters())
            accelerator.print(
                f"[train] DiT: {n_total/1e9:.2f}B total, {n_trainable/1e6:.1f}M trainable "
                f"({100*n_trainable/n_total:.2f}%); action_adapter: {n_aa/1e6:.2f}M "
                f"(legacy variant='v2', no gates); tracks_head: {n_th/1e6:.2f}M  "
                f"lambda_tracks={args.lambda_tracks}  use_tracks_loss={use_tracks_loss}"
            )
        else:
            n_re = sum(p.numel() for p in dit.render_encoder.parameters())
            gate_str = (
                f"render_gate={float(dit.render_gate.detach().float().item()):.4g}"
                if variant == "v1" else "no gate"
            )
            accelerator.print(
                f"[train] DiT: {n_total/1e9:.2f}B total, {n_trainable/1e6:.1f}M trainable "
                f"({100*n_trainable/n_total:.2f}%); render_encoder: {n_re/1e6:.2f}M "
                f"(legacy variant={variant!r}); tracks_head: {n_th/1e6:.2f}M  "
                f"lambda_tracks={args.lambda_tracks}  use_tracks_loss={use_tracks_loss}  "
                f"disable_render_gate={args.disable_render_gate}  {gate_str}"
            )
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    n_samples = len(embed_cache)
    micro_pe = single_gpu_micro_steps_per_epoch(n_samples, args.dataset_repeat)
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
                    f"— raise --num_epochs or dataset_repeat to reach lr floor (epoch>={eed}).  "
                )
            else:
                extra = ""
            extra += (
                f"epoch_decay=[{int(args.lr_epoch_decay_start)},{eed})  "
                f"opt_step_decay=[{int(args.lr_epoch_decay_start) * spep},{s1s})"
            )
        accelerator.print(
            f"[train] lr_scheduler={args.lr_scheduler}  warmup_steps={args.lr_warmup_steps}  "
            f"lr_min_ratio={args.lr_min_ratio}  total_opt_steps={opt_total}  micro_steps/epoch={micro_pe}"
            f"{extra}"
        )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- wandb -------------------------------------------------------------
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

    global_step = 0
    forward_debug_done = False
    # Per-step flow loss varies massively with the random timestep
    # (low/high-sigma extremes are easy, mid-sigma is hard), so a single-step
    # value swings between ~0.3 and ~1.8 even when the model is healthy.
    # We track an EMA + per-quartile-of-timestep stats so the user can see
    # the actual learning trend rather than per-step noise.
    flow_ema: Optional[float] = None
    flow_ema_decay = 0.95
    n_t = len(scheduler.timesteps)
    bucket_edges = [n_t * k // 4 for k in range(5)]
    bucket_sum = [0.0, 0.0, 0.0, 0.0]
    bucket_cnt = [0, 0, 0, 0]
    for epoch in range(args.num_epochs):
        # Reset per-epoch flow-loss accumulator. Within an epoch each rank
        # processes a fixed disjoint slice of samples, so averaging over
        # the full epoch absorbs sample-, timestep- and noise-variance and
        # gives the cleanest trend signal across epochs.
        epoch_flow_sum = 0.0
        epoch_flow_cnt = 0
        indices = torch.randperm(n_samples).tolist()
        if args.dataset_repeat > 1:
            indices = indices * args.dataset_repeat
        bar = tqdm(indices, disable=not accelerator.is_main_process, desc=f"epoch {epoch}")
        for step, sample_idx in enumerate(bar):
            with accelerator.accumulate(dit):
                cached = embed_cache[sample_idx]
                clean_latents = cached["clean_latents"].to(device=device, dtype=dtype)
                render_latents = cached["render_latents"].to(device=device, dtype=dtype)
                prompt_embeds = cached["prompt_embeds"].to(device=device, dtype=dtype)
                image_embeds = cached["image_embeds"].to(device=device, dtype=dtype)
                condition = cached["condition"].to(device=device, dtype=dtype)
                # Optional 8-d (or 7-d) per-frame action stream — fed into the
                # action-aware adapter when the embodiment module's
                # ``use_action_aware_adaln`` is on.
                actions: Optional[torch.Tensor] = None
                if "actions" in cached:
                    actions = cached["actions"].to(
                        device=device, dtype=dtype,
                    ).unsqueeze(0)                                   # (1, T, D)

                # Tracks supervision (optional).
                gt_xy: Optional[torch.Tensor] = None
                gt_vis: Optional[torch.Tensor] = None
                query_xy: Optional[torch.Tensor] = None
                if use_tracks_loss and "gt_xy" in cached:
                    gt_xy = cached["gt_xy"].to(device=device, dtype=torch.float32)
                    gt_vis = cached["gt_vis"].to(device=device, dtype=torch.float32)
                    # add batch dim and (optionally) subsample N query points
                    gt_xy = gt_xy.unsqueeze(0)              # (1, T, N, 2)
                    gt_vis = gt_vis.unsqueeze(0)            # (1, T, N)
                    if args.max_query_points > 0 and gt_xy.shape[2] > args.max_query_points:
                        N_full = gt_xy.shape[2]
                        sel = torch.randperm(N_full, device=device)[: args.max_query_points]
                        gt_xy = gt_xy[:, :, sel]
                        gt_vis = gt_vis[:, :, sel]
                    ref = max(0, min(args.ref_frame, gt_xy.shape[1] - 1))
                    query_xy = gt_xy[:, ref]               # (1, N, 2) in [-1, 1]

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

                # Render-dropout for CFG: see train_fsdp.py for rationale.
                drop_render_step = False
                if args.render_dropout_prob > 0:
                    drop_g = torch.Generator()
                    drop_g.manual_seed(int(args.seed) * 1_000_003 + epoch * 65521 + global_step)
                    drop_render_step = bool(
                        torch.rand((1,), generator=drop_g).item() < args.render_dropout_prob
                    )

                forward_kwargs = dict(
                    hidden_states=latent_model_input,
                    timestep=timestep_batch,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    render_latents=None if drop_render_step else render_latents,
                    return_dict=False,
                )
                if actions is not None and not drop_render_step:
                    forward_kwargs["actions"] = actions
                if query_xy is not None and not drop_render_step:
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
                    ).sum(-1)                              # (1, T, N)
                    vis_sum = gt_vis.sum().clamp(min=1.0)
                    loss_tracks = (diff * gt_vis).sum() / vis_sum
                else:
                    loss_tracks = torch.zeros((), device=device, dtype=loss_flow.dtype)
                loss = loss_flow + args.lambda_tracks * loss_tracks

                # ---- NaN/Inf guard on the loss before backward ----
                loss_is_finite = bool(torch.isfinite(loss.detach()).item())
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
                    if accelerator.sync_gradients:
                        gn = accelerator.clip_grad_norm_(
                            trainable, max_norm=args.max_grad_norm
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
            flow_val = float(loss_flow.detach())
            if flow_ema is None:
                flow_ema = flow_val
            else:
                flow_ema = flow_ema_decay * flow_ema + (1.0 - flow_ema_decay) * flow_val
            t_idx_int = int(idx.item())
            b = min(3, max(0, sum(1 for e in bucket_edges[1:] if t_idx_int >= e)))
            bucket_sum[b] += flow_val
            bucket_cnt[b] += 1
            epoch_flow_sum += flow_val
            epoch_flow_cnt += 1
            if accelerator.is_main_process:
                gn_disp = float(grad_norm) if torch.isfinite(grad_norm).item() else float("nan")
                sigma_frac = float(t_idx_int) / max(1, n_t - 1)
                if step % 5 == 0:
                    bar.set_postfix(
                        loss=f"{float(loss.detach()):.3f}",
                        flow=f"{flow_val:.3f}",
                        ema=f"{flow_ema:.3f}",
                        sig=f"{sigma_frac:.2f}",
                        tr=f"{float(loss_tracks.detach()):.3f}" if pred_tracks is not None else "off",
                        gn=f"{gn_disp:.3f}",
                        ok="Y" if loss_is_finite else "N",
                    )
                if use_wandb and global_step % args.log_every_n_steps == 0:
                    import wandb
                    log_data = {
                        "train/loss": float(loss.detach()),
                        "train/loss_flow": flow_val,
                        "train/loss_flow_ema": float(flow_ema),
                        "train/loss_tracks": float(loss_tracks.detach()),
                        "train/grad_norm": gn_disp,
                        "train/loss_is_finite": float(loss_is_finite),
                        "train/lambda_tracks": args.lambda_tracks,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/timestep": float(timestep_scalar.item()),
                        "train/timestep_sigma_frac": sigma_frac,
                        "train/epoch": epoch,
                    }
                    for k in range(4):
                        if bucket_cnt[k] > 0:
                            log_data[f"train/flow_by_t_q{k}"] = bucket_sum[k] / bucket_cnt[k]
                    wandb.log(log_data, step=global_step)

        # End-of-epoch summary: mean flow loss over the entire epoch is the
        # cleanest "is loss going down?" curve.
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

        if accelerator.is_main_process:
            if int(args.condition_usage_sanity_samples) > 0:
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

            save_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(dit)
            if args.trainable == "render_only":
                # Filter the FULL state_dict by trainable-key prefix. Mirrors
                # the train_fsdp.py save logic (keep_prefixes), so train.py
                # and train_fsdp.py produce identical checkpoint layouts and
                # any code path can load either. Centralizes which keys are
                # "render_only-trainable" so we don't have to keep model-side
                # adds in sync with this save block.
                full_state = unwrapped.state_dict()
                if args.use_embodiment_adapter:
                    keep_prefixes = ["embodiment."]
                    keep_exact: set = set()
                elif getattr(unwrapped, "legacy_render_variant", "egowm") == "v2":
                    # v2: ActionConditionedTembAdapter at `action_adapter.*`.
                    keep_prefixes = ["action_adapter."]
                    keep_exact = set()
                else:
                    keep_prefixes = ["render_encoder."]
                    keep_exact = set()
                    if getattr(unwrapped, "legacy_render_variant", "egowm") == "v1":
                        keep_prefixes.append("render_fuse.")
                        keep_exact.add("render_gate")
                if use_tracks_loss:
                    keep_prefixes.append("tracks_head.")
                if getattr(args, "unfreeze_last_n_blocks", 0) > 0:
                    n_blocks = args.unfreeze_last_n_blocks
                    n_total_blocks = len(unwrapped.blocks)
                    for i in range(n_total_blocks - n_blocks, n_total_blocks):
                        keep_prefixes.append(f"blocks.{i}.")
                    keep_prefixes.append("norm_out.")
                    keep_prefixes.append("proj_out.")
                    keep_prefixes.append("condition_embedder.time_proj.")
                kp = tuple(keep_prefixes)
                render_state = {
                    k: v.detach().cpu()
                    for k, v in full_state.items()
                    if k.startswith(kp) or k in keep_exact
                }
                ckpt_path = os.path.join(save_dir, "render_conditioner.pt")
                torch.save(render_state, ckpt_path)
                n_saved = sum(v.numel() for v in render_state.values())
                accelerator.print(
                    f"Saved render conditioner ({n_saved/1e6:.1f}M params, "
                    f"{len(render_state)} tensors) to {ckpt_path}"
                )
            else:
                unwrapped.save_pretrained(save_dir)
                accelerator.print(f"Saved transformer to {save_dir}")

    if accelerator.is_main_process and use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
