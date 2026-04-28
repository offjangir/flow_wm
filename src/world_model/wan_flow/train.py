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

        noise = torch.randn_like(clean)
        t_idx = torch.randint(0, len(scheduler.timesteps), (1,), generator=g)
        t_scalar = scheduler.timesteps[t_idx].to(device=device).float()
        t_batch = t_scalar.expand(clean.shape[0])
        noisy = scheduler.scale_noise(clean, t_batch, noise)
        noisy[:, :, 0:1] = clean[:, :, 0:1]

        latent_in = torch.cat([noisy, cond], dim=1)
        target = noise - clean

        pred_right = model(
            hidden_states=latent_in,
            timestep=t_batch,
            encoder_hidden_states=prompt,
            encoder_hidden_states_image=image,
            render_latents=rr,
            return_dict=False,
        )[0]
        pred_wrong = model(
            hidden_states=latent_in,
            timestep=t_batch,
            encoder_hidden_states=prompt,
            encoder_hidden_states_image=image,
            render_latents=rw,
            return_dict=False,
        )[0]
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
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()
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
    p.add_argument("--output_dir", type=str, default="./checkpoints/wan_render")
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
    if defaults:
        p.set_defaults(**{k: v for k, v in defaults.items() if k != "config"})
    args = p.parse_args(remaining)
    if not args.model_path or not args.dataset_base_path or not args.metadata_csv:
        p.error("Provide --model_path, --dataset_base_path, and --metadata_csv (via CLI or --config JSON).")
    return args


def _precompute_embeddings(
    model_path, dataset, device, height, width, num_frames, max_seq_len,
    encoder_dtype: torch.dtype = torch.bfloat16,
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
    """
    import gc
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.video_processor import VideoProcessor
    from transformers import AutoImageProcessor, AutoTokenizer, CLIPVisionModel, UMT5EncoderModel
    from world_model.wan_flow.data import _load_tracks_npz, _load_video_frames
    from world_model.wan_flow.model import WanVAEChunkedEncode

    root = model_path.rstrip("/")
    rows = dataset.rows
    n = len(rows)
    dev_str = str(device)

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

    print(f"[precompute 1/3] text encoder → GPU, encoding {n} prompts ...")
    tokenizer = AutoTokenizer.from_pretrained(root, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        root, subfolder="text_encoder", torch_dtype=encoder_dtype,
        device_map=dev_str, low_cpu_mem_usage=True,
    )
    text_encoder.eval()
    all_prompt_embeds = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            tok = tokenizer(
                [str(row["prompt"])],
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
    img_proc = AutoImageProcessor.from_pretrained(root, subfolder="image_processor")
    image_encoder = CLIPVisionModel.from_pretrained(
        root, subfolder="image_encoder", torch_dtype=encoder_dtype,
        device_map=dev_str, low_cpu_mem_usage=True,
    )
    image_encoder.eval()
    all_image_embeds = []
    with torch.no_grad():
        for i, row in enumerate(rows):
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
        for i, row in enumerate(rows):
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
            trajs, visibs, _img = _load_tracks_npz(tpath, num_frames)
            all_tracks.append({"gt_xy": trajs, "gt_vis": visibs})
        n_with = sum(1 for t in all_tracks if t is not None)
        print(f"[precompute tracks] {n_with}/{n} samples have tracks")
    else:
        all_tracks = [None] * n

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
        cache.append(sample)
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
    tkw = dict(torch_dtype=dit_load_dtype, render_encoder_kwargs={})
    if local_transformer:
        dit = WanTransformerRenderConditioned.from_pretrained(os.path.join(root, "transformer"), **tkw)
    else:
        dit = WanTransformerRenderConditioned.from_pretrained(root, subfolder="transformer", **tkw)

    _materialize_meta_submodules(dit)
    dit = dit.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    dit.train()
    dit.enable_gradient_checkpointing()

    use_tracks_loss = args.lambda_tracks > 0 and any("gt_xy" in c for c in embed_cache)
    if args.trainable == "render_only":
        for p_ in dit.parameters():
            p_.requires_grad = False
        for p_ in dit.render_encoder.parameters():
            p_.requires_grad = True
        for p_ in dit.render_fuse.parameters():
            p_.requires_grad = True
        dit.render_gate.requires_grad = True
        if use_tracks_loss:
            for p_ in dit.tracks_head.parameters():
                p_.requires_grad = True
    else:
        for p_ in dit.parameters():
            p_.requires_grad = True

    if args.disable_render_gate:
        with torch.no_grad():
            dit.render_gate.data.fill_(1.0)
        dit.render_gate.requires_grad = False

    trainable = [p for p in dit.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in dit.parameters())
    n_th = sum(p.numel() for p in dit.tracks_head.parameters())
    accelerator.print(
        f"[train] DiT: {n_total/1e9:.2f}B total, {n_trainable/1e6:.1f}M trainable "
        f"({100*n_trainable/n_total:.2f}%); tracks_head: {n_th/1e6:.2f}M  "
        f"lambda_tracks={args.lambda_tracks}  use_tracks_loss={use_tracks_loss}  "
        f"disable_render_gate={args.disable_render_gate}  render_gate={float(dit.render_gate.detach().float().item()):.4g}"
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

                forward_kwargs = dict(
                    hidden_states=latent_model_input,
                    timestep=timestep_batch,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    render_latents=render_latents,
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
                render_state = {
                    **{f"render_encoder.{k}": v.detach().cpu()
                       for k, v in unwrapped.render_encoder.state_dict().items()},
                    **{f"render_fuse.{k}": v.detach().cpu()
                       for k, v in unwrapped.render_fuse.state_dict().items()},
                    "render_gate": unwrapped.render_gate.detach().cpu(),
                }
                if use_tracks_loss:
                    render_state.update({
                        f"tracks_head.{k}": v.detach().cpu()
                        for k, v in unwrapped.tracks_head.state_dict().items()
                    })
                ckpt_path = os.path.join(save_dir, "render_conditioner.pt")
                torch.save(render_state, ckpt_path)
                n_saved = sum(v.numel() for v in render_state.values())
                accelerator.print(
                    f"Saved render conditioner ({n_saved/1e6:.1f}M params) to {ckpt_path}"
                )
            else:
                unwrapped.save_pretrained(save_dir)
                accelerator.print(f"Saved transformer to {save_dir}")

    if accelerator.is_main_process and use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
