#!/usr/bin/env python3
"""
Evaluate the render-conditioned Wan 2.1 I2V world model on the training
dataset (or any metadata.csv with the same schema).

For each chosen sample, this script:

  1. Loads the first frame of the real DROID camera video (``video``).
  2. VAE-encodes the corresponding DrRobot render video (``render``) to
     get the per-latent-frame action conditioning.
  3. Runs ``RenderConditionedWanI2VPipeline`` to generate a predicted
     video conditioned on (first_frame, prompt, render_video).
  4. Writes a side-by-side comparison MP4:
        [ ground-truth real | DrRobot render | model prediction ]
     so you can visually check what the model is actually predicting.

Checkpoint discovery: by default, picks the highest ``epoch_*`` directory
under ``--checkpoint_dir`` and loads its ``render_conditioner.pt`` (the
small render_encoder + render_fuse [+ render_gate] [+ tracks_head] subset
saved during training). Pre-fix checkpoints that lack ``render_gate`` are
loaded with ``strict=False`` -- the in-model gate stays at its zero init
so the model degrades to plain pretrained Wan I2V on those samples,
which is itself a useful sanity baseline.

Usage::

  python scripts/eval_world_model.py \
      --model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
      --checkpoint_dir ./checkpoints/wan_render_fsdp \
      --metadata_csv ./data_wan/metadata.csv \
      --dataset_base_path ./data_wan \
      --output_dir ./eval_outputs \
      --num_samples 3 \
      --num_inference_steps 50 \
      --num_frames 33 --height 320 --width 576

Run on a single GPU (no FSDP / accelerate). Memory: ~30 GB peak in bf16.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from typing import List, Optional

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image

from world_model.wan_flow.model import (
    RenderConditionedWanI2VPipeline,
    WanTransformerRenderConditioned,
)
from world_model.wan_flow.train import _materialize_meta_submodules

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import CLIPVisionModel


# ----------------------------------------------------------------------- args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference with the render-conditioned Wan I2V world model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", type=str, default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                   help="HF id or local Diffusers root for Wan I2V (with transformer/, vae/, ...).")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/wan_render_fsdp",
                   help="Directory containing epoch_<N>/render_conditioner.pt subdirs. Latest is picked unless --checkpoint_path is set.")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Explicit path to a render_conditioner.pt (overrides --checkpoint_dir).")
    p.add_argument("--metadata_csv", type=str, default="./data_wan/metadata.csv")
    p.add_argument("--dataset_base_path", type=str, default="./data_wan")
    p.add_argument("--output_dir", type=str, default="./eval_outputs")

    p.add_argument("--num_samples", type=int, default=3,
                   help="How many rows from the CSV to evaluate (in CSV order).")
    p.add_argument("--sample_indices", type=str, default=None,
                   help="Comma-separated CSV row indices (0-based). Overrides --num_samples.")

    p.add_argument("--num_frames", type=int, default=33,
                   help="Number of generated frames. Must satisfy (num_frames - 1) %% 4 == 0 for Wan VAE.")
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--max_sequence_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--mixed_precision", choices=("bf16", "fp16", "no"), default="bf16",
                   help="Inference dtype. Defaults to bf16 to match the FSDP training storage dtype.")
    p.add_argument("--fps", type=int, default=8, help="Output MP4 framerate.")
    p.add_argument("--no_render_conditioner", action="store_true",
                   help="Skip loading render_conditioner.pt -- evaluate plain pretrained Wan I2V (sanity baseline).")
    p.add_argument(
        "--render_gate_override",
        type=float,
        default=None,
        help="If set, force transformer.render_gate to this value after loading checkpoint "
             "(e.g. 1.0 to fully enable render conditioning for ablation).",
    )
    p.add_argument("--cpu_offload", action=argparse.BooleanOptionalAction, default=True,
                   help="Use diffusers' model CPU offload (each submodule lives on CPU "
                        "and is moved to GPU only for its forward). Required on a single "
                        "48 GB card; pass --no-cpu_offload on H100/A100-80GB for max speed.")
    p.add_argument("--force_empty_prompt", action="store_true",
                   help="Ignore the CSV `prompt` column and feed an empty string to the text "
                        "encoder. Use this when the checkpoint was trained with "
                        "`ignore_prompts: true` (text encoder always saw \"\" during training).")
    # ---- v2 action-conditioned path ----
    p.add_argument("--action_stats_path", type=str, default=None,
                   help="z-score stats JSON for action normalization. Required for v2 "
                        "checkpoints trained with action_stats_path set; pass the same "
                        "file that was used at training time.")
    p.add_argument("--action_field", type=str, default="state",
                   help="Field name in actions/<scene>.npz. 'state' = 8-d joint+gripper "
                        "(default); 'action' = 7-d commanded target+gripper.")
    p.add_argument("--legacy_render_variant", type=str, default="egowm",
                   choices=("egowm", "v1", "v2", "v3", "v4"),
                   help="Selects the transformer subclass. v3/v4 require their "
                        "respective adapter_kwargs JSON to match training.")
    p.add_argument("--v3_adapter_kwargs", type=json.loads, default=None,
                   help="JSON dict of v3 adapter kwargs (must match training args.json).")
    p.add_argument("--v4_adapter_kwargs", type=json.loads, default=None,
                   help="JSON dict of v4 adapter kwargs (must match training args.json).")
    return p.parse_args()


# ------------------------------------------------------ checkpoint discovery


def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return path to the highest-epoch ``render_conditioner.pt`` under ``ckpt_dir``."""
    if not os.path.isdir(ckpt_dir):
        return None
    pat = re.compile(r"^epoch_(\d+)$")
    best = (-1, None)
    for name in os.listdir(ckpt_dir):
        m = pat.match(name)
        if not m:
            continue
        cand = os.path.join(ckpt_dir, name, "render_conditioner.pt")
        if not os.path.isfile(cand):
            continue
        e = int(m.group(1))
        if e > best[0]:
            best = (e, cand)
    return best[1]


# ----------------------------------------------------------- I/O utilities


def _load_video_frames(path: str, num_frames: int) -> List[Image.Image]:
    """Read MP4 -> uniformly subsample to num_frames PIL images (RGB)."""
    reader = imageio.get_reader(path, "ffmpeg")
    try:
        all_frames = [Image.fromarray(fr).convert("RGB") for fr in reader]
    finally:
        reader.close()
    if not all_frames:
        raise ValueError(f"No frames decoded from {path}")
    idxs = np.linspace(0, len(all_frames) - 1, num_frames).astype(int)
    return [all_frames[i] for i in idxs]


def _resize_pils(frames: List[Image.Image], height: int, width: int) -> List[Image.Image]:
    return [f.resize((width, height), Image.BILINEAR) for f in frames]


def _frames_to_uint8(frames: List[np.ndarray] | np.ndarray) -> np.ndarray:
    """Coerce a list of HxWx3 frames (or stacked TxHxWx3 array) in [0,1] or
    [0,255] to a uint8 ndarray of shape (T, H, W, 3)."""
    arr = np.asarray(frames)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        pass
    elif arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    else:
        raise ValueError(f"Unexpected video array shape {arr.shape}")
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _hstack_rows(*videos_uint8: np.ndarray, gap: int = 4) -> np.ndarray:
    """Horizontally stack videos with a black gap. All must have the same T,H."""
    if not videos_uint8:
        raise ValueError("need at least one video")
    T = videos_uint8[0].shape[0]
    H = videos_uint8[0].shape[1]
    for v in videos_uint8:
        if v.shape[0] != T or v.shape[1] != H:
            raise ValueError("all videos must share T and H to hstack")
    spacer = np.zeros((T, H, gap, 3), dtype=np.uint8)
    pieces: List[np.ndarray] = []
    for i, v in enumerate(videos_uint8):
        if i > 0:
            pieces.append(spacer)
        pieces.append(v)
    return np.concatenate(pieces, axis=2)  # along width


def _per_frame_psnr_ssim(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Compute mean PSNR/SSIM/MSE between two (T,H,W,3) uint8 arrays. Per-frame, then averaged.

    SSIM uses channel_axis=-1 (RGB), data_range=255. PSNR uses data_range=255.
    MSE is reported in normalized [0,1]² pixel space (uint8 / 255.0) so it's
    invariant to the integer encoding and directly comparable to losses
    reported in normalized space.
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    assert pred.shape == ref.shape, f"shape mismatch: {pred.shape} vs {ref.shape}"
    T = pred.shape[0]
    psnr_per = np.empty(T, dtype=np.float64)
    ssim_per = np.empty(T, dtype=np.float64)
    mse_per = np.empty(T, dtype=np.float64)
    pred_n = pred.astype(np.float64) / 255.0
    ref_n = ref.astype(np.float64) / 255.0
    for t in range(T):
        psnr_per[t] = peak_signal_noise_ratio(ref[t], pred[t], data_range=255)
        ssim_per[t] = structural_similarity(
            ref[t], pred[t], data_range=255, channel_axis=-1
        )
        mse_per[t] = float(np.mean((pred_n[t] - ref_n[t]) ** 2))
    return {
        "psnr_per_frame": psnr_per.tolist(),
        "ssim_per_frame": ssim_per.tolist(),
        "mse_per_frame": mse_per.tolist(),
        "psnr_mean": float(psnr_per.mean()),
        "ssim_mean": float(ssim_per.mean()),
        "mse_mean": float(mse_per.mean()),
        "psnr_frame0": float(psnr_per[0]),
        "ssim_frame0": float(ssim_per[0]),
        "mse_frame0": float(mse_per[0]),
        "psnr_last": float(psnr_per[-1]),
        "ssim_last": float(ssim_per[-1]),
        "mse_last": float(mse_per[-1]),
    }


def _label_frames(frames: np.ndarray, label: str, band_h: int = 24) -> np.ndarray:
    """Add a black band with white text at the top of each frame.

    Uses PIL.ImageDraw with the default bitmap font (no extra deps)."""
    from PIL import ImageDraw

    T, H, W, _ = frames.shape
    out = np.zeros((T, H + band_h, W, 3), dtype=np.uint8)
    out[:, band_h:] = frames
    img = Image.new("RGB", (W, band_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((6, 4), label, fill=(255, 255, 255))
    band = np.array(img)
    out[:, :band_h] = band[None]
    return out


# -------------------------------------------------------- model construction


def _build_pipeline(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> RenderConditionedWanI2VPipeline:
    root = args.model_path.rstrip("/")
    local_transformer = os.path.isdir(os.path.join(root, "transformer"))
    variant = getattr(args, "legacy_render_variant", "egowm")
    print(f"[eval] loading transformer (render-conditioned, variant={variant}) from {root} ...")
    tkw = dict(torch_dtype=dtype, render_encoder_kwargs={})
    if variant == "v3":
        from world_model.wan_flow.model_v3 import WanTransformerRenderConditionedV3
        model_cls = WanTransformerRenderConditionedV3
        if getattr(args, "v3_adapter_kwargs", None) is not None:
            tkw["v3_adapter_kwargs"] = args.v3_adapter_kwargs
    elif variant == "v4":
        from world_model.wan_flow.model_v4 import WanTransformerRenderConditionedV4
        model_cls = WanTransformerRenderConditionedV4
        if getattr(args, "v4_adapter_kwargs", None) is not None:
            tkw["v4_adapter_kwargs"] = args.v4_adapter_kwargs
    else:
        model_cls = WanTransformerRenderConditioned
        tkw["legacy_render_variant"] = variant
    if local_transformer:
        transformer = model_cls.from_pretrained(os.path.join(root, "transformer"), **tkw)
    else:
        transformer = model_cls.from_pretrained(root, subfolder="transformer", **tkw)
    # `from_pretrained` leaves newly-added modules (render_encoder, render_fuse,
    # render_gate, tracks_head) on the meta device because they don't exist
    # in the pretrained Wan I2V safetensors. Materialize them on CPU before the
    # `pipe.to(device)` call later, otherwise PyTorch raises
    # "Cannot copy out of meta tensor; no data!". `_materialize_meta_submodules`
    # uses each module's own `reset_parameters` (or zero-init for nn.Parameters
    # like render_gate) to give them real values; the trained weights, if any,
    # get loaded on top via `_maybe_load_render_conditioner` after this.
    _materialize_meta_submodules(transformer)

    print(f"[eval] loading VAE + image encoder ...")
    vae = AutoencoderKLWan.from_pretrained(root, subfolder="vae", torch_dtype=dtype)
    image_encoder = CLIPVisionModel.from_pretrained(
        root, subfolder="image_encoder", torch_dtype=torch.float32
    )

    print(f"[eval] assembling pipeline ...")
    pipe = RenderConditionedWanI2VPipeline.from_pretrained(
        root,
        transformer=transformer,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    # Replace the scheduler with a fresh FlowMatchEuler one (matches training).
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")
    # Memory budget on a single 48 GB card is tight: 14 B params (28 GB bf16)
    # + VAE/CLIP/T5 (~5 GB) + per-block per-token AdaLN modulation tensors
    # (Eq. 5 makes them ~10x bigger than vanilla Wan) easily clears 47 GB.
    # Training fits because FSDP across 4 GPUs shards the 28 GB of params and
    # frees per-layer activations between blocks; eval on a single GPU has no
    # such relief.
    #
    # Two lines of defense:
    #  (1) VAE tiling/slicing so the conditioning encode/decode streams tiles
    #      instead of allocating the full activation map at once.
    #  (2) Model CPU offload: each top-level submodule (text encoder, image
    #      encoder, transformer, vae) lives on CPU and is brought onto GPU
    #      only while running. Roughly 2x slower but fits in <8 GB headroom.
    #      Skipped if the user passed --no_offload (e.g. when running on a
    #      large-memory GPU like H100/A100-80GB where the full pipeline fits).
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if args.cpu_offload:
        # `enable_model_cpu_offload` itself moves the pipeline to `device`,
        # so we skip the manual `pipe.to(device)` below.
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    return pipe


def _maybe_load_render_conditioner(
    pipe: RenderConditionedWanI2VPipeline,
    ckpt_path: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Load the small render_conditioner state dict into pipe.transformer.

    Returns a metadata dict (n_keys_loaded, n_keys_expected, has_render_gate, ...).
    """
    if ckpt_path is None:
        return {"loaded": False, "reason": "no checkpoint provided (running pretrained baseline)."}
    print(f"[eval] loading render conditioner from {ckpt_path} ...")
    sd = torch.load(ckpt_path, map_location="cpu")
    # Cast loaded tensors to model dtype (storage in ckpt may be bf16/fp32 depending on save path).
    sd = {k: v.to(dtype=dtype) if v.is_floating_point() else v for k, v in sd.items()}
    has_gate = "render_gate" in sd
    missing, unexpected = pipe.transformer.load_state_dict(sd, strict=False)
    # ``strict=False`` returns lists naming the keys that didn't get loaded
    # from `sd` AND keys in the model that weren't in `sd`. We want to flag
    # the latter loudly, but only for the render_* / tracks_head subset --
    # the frozen Wan blocks are SUPPOSED to be missing from the small ckpt.
    relevant_missing = [
        k for k in missing
        if k.startswith(("render_encoder.", "render_fuse.", "tracks_head."))
        or k == "render_gate"
    ]
    return {
        "loaded": True,
        "ckpt_path": ckpt_path,
        "n_keys_in_ckpt": len(sd),
        "has_render_gate": has_gate,
        "render_gate_value": float(sd["render_gate"].float().item()) if has_gate else 0.0,
        "missing_render_keys": relevant_missing,
        "unexpected_keys": list(unexpected),
    }


# ---------------------------------------------------------------- main


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (
        torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )

    if args.num_frames % 4 != 1:
        raise ValueError(
            f"--num_frames must satisfy (num_frames - 1) % 4 == 0 for the Wan VAE; got {args.num_frames}."
        )

    df = pd.read_csv(args.metadata_csv)
    required = {"video", "render", "prompt"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"metadata CSV missing columns: {missing_cols}")

    if args.sample_indices:
        sel = [int(s.strip()) for s in args.sample_indices.split(",") if s.strip()]
    else:
        sel = list(range(min(args.num_samples, len(df))))
    if not sel:
        raise ValueError("No samples selected.")
    print(f"[eval] evaluating {len(sel)} sample(s) at indices {sel} from {args.metadata_csv}")

    base = os.path.abspath(args.dataset_base_path)
    def _abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(base, p)

    pipe = _build_pipeline(args, device, dtype)

    if args.no_render_conditioner:
        ckpt_meta = {"loaded": False, "reason": "--no_render_conditioner"}
    else:
        ckpt_path = args.checkpoint_path or _find_latest_checkpoint(args.checkpoint_dir)
        if ckpt_path is None:
            print(f"[eval] WARNING: no render_conditioner.pt found under {args.checkpoint_dir}; "
                  "running plain pretrained Wan I2V (the render_gate stays at 0).")
        ckpt_meta = _maybe_load_render_conditioner(pipe, ckpt_path, device, dtype)
    if args.render_gate_override is not None:
        with torch.no_grad():
            pipe.transformer.render_gate.data.fill_(float(args.render_gate_override))
        ckpt_meta["render_gate_overridden"] = True
        ckpt_meta["render_gate_value_after_override"] = float(pipe.transformer.render_gate.float().item())
    print(f"[eval] checkpoint meta: {json.dumps(ckpt_meta, indent=2, default=str)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve action stats once (z-score normalization).
    # Needed for v2 (legacy_render_variant='v2') checkpoints; harmless for egowm
    # (no `actions` ever read from cache).
    is_v2 = (
        getattr(pipe.transformer, "legacy_render_variant", None) == "v2"
        or hasattr(pipe.transformer, "action_adapter")
    )
    action_stats = None
    if is_v2 and args.action_stats_path:
        from world_model.wan_flow.data import _load_action_stats
        action_stats = _load_action_stats(args.action_stats_path)
        print(f"[eval] loaded action stats from {args.action_stats_path}")
    if is_v2 and "actions" not in df.columns:
        raise SystemExit(
            "v2 checkpoint requires the metadata CSV to have an 'actions' column "
            "pointing to per-scene .npz files; not found in "
            f"{args.metadata_csv}."
        )

    summary = []
    for k, idx in enumerate(sel):
        row = df.iloc[idx].to_dict()
        vpath = _abs(row["video"])
        rpath = _abs(row["render"])
        if args.force_empty_prompt:
            prompt = ""
        else:
            prompt_value = row.get("prompt", "")
            prompt = "" if pd.isna(prompt_value) else str(prompt_value)
        scene = os.path.splitext(os.path.basename(vpath))[0]
        print(f"\n[eval] [{k+1}/{len(sel)}] scene={scene}")
        print(f"        video  = {vpath}")
        print(f"        render = {rpath}")
        print(f"        prompt = {prompt!r}")
        if hasattr(pipe.transformer, "render_gate"):
            print(f"        render_gate = {float(pipe.transformer.render_gate.item()):.6f}")

        gt_frames = _resize_pils(_load_video_frames(vpath, args.num_frames), args.height, args.width)
        render_frames = _resize_pils(_load_video_frames(rpath, args.num_frames), args.height, args.width)
        first_frame = gt_frames[0]

        # Load per-scene actions for v2. The cross-attn KV dim is num_frames.
        actions_tensor: Optional[torch.Tensor] = None
        if is_v2:
            from world_model.wan_flow.data import _load_actions_npz
            apath_raw = row.get("actions", "")
            if pd.isna(apath_raw) or not str(apath_raw):
                raise SystemExit(
                    f"row {idx}: 'actions' column empty/NaN but checkpoint is v2."
                )
            apath = _abs(str(apath_raw))
            actions_np = _load_actions_npz(
                apath, args.num_frames, stats=action_stats, field=args.action_field,
            )
            actions_tensor = torch.from_numpy(actions_np).unsqueeze(0).to(
                device=device, dtype=dtype,
            )
            print(f"        actions = {apath}  shape={tuple(actions_tensor.shape)}  "
                  f"field={args.action_field}  stats={'z-score' if action_stats is not None else 'raw'}")

        gen = torch.Generator(device=device).manual_seed(args.seed + idx)
        t0 = time.time()
        # The base Wan 2.1 transformer is loaded mostly in bf16, but a couple of
        # tiny submodules (notably `condition_embedder.time_proj`) keep fp32
        # weights -- this is how the official safetensors are stored. The base
        # `WanTimeTextImageEmbedding.forward` casts `temb` to the (bf16) text
        # embedding dtype right before the fp32 `time_proj` linear, which raises
        # `mat1 and mat2 must have the same dtype` outside of autocast. The
        # training scripts work because `accelerator.autocast()` wraps the step;
        # inference has no autocast by default, so we enable it explicitly here
        # for bf16 / fp16. fp32 evaluation skips the wrapper.
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
            if dtype in (torch.bfloat16, torch.float16)
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            out = pipe(
                image=first_frame,
                prompt=prompt,
                negative_prompt="",
                render_video=render_frames,
                actions=actions_tensor,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                max_sequence_length=args.max_sequence_length,
                generator=gen,
                output_type="np",
                return_dict=False,
            )
        elapsed = time.time() - t0
        pred_frames = _frames_to_uint8(out[0])
        gt_arr = _frames_to_uint8(np.stack([np.asarray(f) for f in gt_frames]))
        rd_arr = _frames_to_uint8(np.stack([np.asarray(f) for f in render_frames]))

        if pred_frames.shape[0] != gt_arr.shape[0]:
            T = min(pred_frames.shape[0], gt_arr.shape[0], rd_arr.shape[0])
            pred_frames = pred_frames[:T]
            gt_arr = gt_arr[:T]
            rd_arr = rd_arr[:T]

        # Quality metrics: pred vs GT (real DROID), and pred vs render (DrRobot).
        # First-frame metrics should be near-perfect (pipeline pins frame 0 to GT);
        # the trajectory across frames is what shows whether the model is rolling
        # out the dynamics rather than just regurgitating the start frame.
        metrics_vs_gt = _per_frame_psnr_ssim(pred_frames, gt_arr)
        metrics_vs_render = _per_frame_psnr_ssim(pred_frames, rd_arr)
        print(
            f"        psnr vs GT: mean={metrics_vs_gt['psnr_mean']:.2f}dB  "
            f"f0={metrics_vs_gt['psnr_frame0']:.2f}  fT={metrics_vs_gt['psnr_last']:.2f}  "
            f"| ssim={metrics_vs_gt['ssim_mean']:.4f} (f0={metrics_vs_gt['ssim_frame0']:.4f}, fT={metrics_vs_gt['ssim_last']:.4f})  "
            f"| mse={metrics_vs_gt['mse_mean']:.5f} (f0={metrics_vs_gt['mse_frame0']:.5f}, fT={metrics_vs_gt['mse_last']:.5f})"
        )
        print(
            f"        psnr vs render: mean={metrics_vs_render['psnr_mean']:.2f}dB  "
            f"| ssim={metrics_vs_render['ssim_mean']:.4f}  "
            f"| mse={metrics_vs_render['mse_mean']:.5f}"
        )

        gt_arr   = _label_frames(gt_arr,   "GT (real DROID)")
        rd_arr   = _label_frames(rd_arr,   "render (DrRobot)")
        pred_arr = _label_frames(
            pred_frames,
            f"prediction  PSNR={metrics_vs_gt['psnr_mean']:.1f}  SSIM={metrics_vs_gt['ssim_mean']:.3f}  MSE={metrics_vs_gt['mse_mean']:.4f}",
        )

        stack = _hstack_rows(gt_arr, rd_arr, pred_arr)
        out_name = f"eval_idx{idx:03d}_{scene}.mp4"
        out_path = os.path.join(args.output_dir, out_name)
        writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264",
                                    pixelformat="yuv420p", macro_block_size=None)
        try:
            for f in stack:
                writer.append_data(f)
        finally:
            writer.close()
        print(f"        wrote {out_path}  ({elapsed:.1f}s, {stack.shape})")
        summary.append({
            "csv_idx": idx,
            "scene": scene,
            "elapsed_s": round(elapsed, 2),
            "out_path": out_path,
            "render_gate": float(pipe.transformer.render_gate.item()) if hasattr(pipe.transformer, "render_gate") else None,
            "metrics_vs_gt": metrics_vs_gt,
            "metrics_vs_render": metrics_vs_render,
        })

    # Aggregate metrics across samples.
    agg = {}
    if summary and "metrics_vs_gt" in summary[0]:
        agg = {
            "psnr_vs_gt_mean":     float(np.mean([s["metrics_vs_gt"]["psnr_mean"] for s in summary])),
            "ssim_vs_gt_mean":     float(np.mean([s["metrics_vs_gt"]["ssim_mean"] for s in summary])),
            "mse_vs_gt_mean":      float(np.mean([s["metrics_vs_gt"]["mse_mean"]  for s in summary])),
            "psnr_vs_render_mean": float(np.mean([s["metrics_vs_render"]["psnr_mean"] for s in summary])),
            "ssim_vs_render_mean": float(np.mean([s["metrics_vs_render"]["ssim_mean"] for s in summary])),
            "mse_vs_render_mean":  float(np.mean([s["metrics_vs_render"]["mse_mean"]  for s in summary])),
            # Per-frame mean across samples — same length T as the predictions.
            # Useful for plotting "how does MSE/PSNR/SSIM degrade across the rollout?".
            "psnr_vs_gt_per_frame_mean": np.mean(
                np.stack([s["metrics_vs_gt"]["psnr_per_frame"] for s in summary]), axis=0
            ).tolist(),
            "ssim_vs_gt_per_frame_mean": np.mean(
                np.stack([s["metrics_vs_gt"]["ssim_per_frame"] for s in summary]), axis=0
            ).tolist(),
            "mse_vs_gt_per_frame_mean": np.mean(
                np.stack([s["metrics_vs_gt"]["mse_per_frame"] for s in summary]), axis=0
            ).tolist(),
            "n_samples": len(summary),
        }
        print(
            f"\n[eval] aggregate over {agg['n_samples']} samples: "
            f"PSNR vs GT={agg['psnr_vs_gt_mean']:.2f}dB  SSIM vs GT={agg['ssim_vs_gt_mean']:.4f}  MSE vs GT={agg['mse_vs_gt_mean']:.5f}  "
            f"| PSNR vs render={agg['psnr_vs_render_mean']:.2f}dB  SSIM vs render={agg['ssim_vs_render_mean']:.4f}  MSE vs render={agg['mse_vs_render_mean']:.5f}"
        )

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": ckpt_meta,
            "args": vars(args),
            "aggregate_metrics": agg,
            "samples": summary,
        }, f, indent=2, default=str)
    print(f"\n[eval] wrote summary -> {summary_path}")
    print(f"[eval] done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
