#!/usr/bin/env python3
"""
Generate videos from a trained render-conditioned Wan I2V checkpoint.

Single-GPU. For each held-out sample, runs the full denoising loop with the
correct render condition and (optionally) ablations:

  - ``generated_<scene>.mp4``       — model with the CORRECT render condition
  - ``generated_no_render_<scene>.mp4``  — render_latents=None (vanilla I2V)
  - ``generated_wrong_render_<scene>.mp4`` — SHUFFLED render (should differ)
  - ``real_<scene>.mp4``            — copy/symlink of the ground-truth DROID mp4
  - ``render_<scene>.mp4``          — copy/symlink of the DrRobot render mp4

Visual inspection of these side-by-side is the actual "does the model use
conditioning" test (eval_render_conditioning.py only measures flow MSE).

Usage::

  python scripts/generate_videos.py \\
    --config configs/train_drrobot_1k_legacy_8xl40.json \\
    --ckpt   checkpoints/wan_render_drrobot_1k_legacy_8xl40/run_.../epoch_9/render_conditioner.pt \\
    --eval_csv data_wan_eval/eval_metadata.csv \\
    --dataset_base_path data_wan_eval \\
    --num_inference_steps 30 \\
    --num_samples 3 \\
    --do_ablations \\
    --out_dir generations_epoch_9
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import imageio_ffmpeg

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _p in (SRC, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from world_model.wan_flow.data import RenderI2VMetadataDataset            # noqa: E402
from world_model.wan_flow.model import WanVAEChunkedEncode                # noqa: E402
from world_model.wan_flow.train import (                                   # noqa: E402
    _materialize_meta_submodules,
    _precompute_embeddings,
)
from world_model.wan_flow.train_fsdp import _load_dit_only                 # noqa: E402
from _eval_common import (                                                  # noqa: E402
    compute_psnr_ssim,
    verify_eval_matches_training,
)


def _load_ckpt_subset(model: torch.nn.Module, ckpt_path: str) -> None:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    target_dtype = next(model.parameters()).dtype
    sd_cast = {
        k: (v.to(dtype=target_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
        for k, v in sd.items()
    }
    missing, unexpected = model.load_state_dict(sd_cast, strict=False)
    print(f"[ckpt] loaded {len(sd_cast) - len(unexpected)}/{len(sd_cast)} tensors from {ckpt_path}")
    if unexpected:
        print(f"[ckpt] WARN unexpected keys (first 5): {unexpected[:5]}")


@torch.no_grad()
def _denoise(
    model: torch.nn.Module,
    scheduler,
    clean_latents_for_first_frame: torch.Tensor,
    cached: dict,
    render_latents: Optional[torch.Tensor],
    num_inference_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """Run the flow-matching denoising loop. Returns final latents (1,16,T,H,W).

    ``cfg_scale > 1.0`` enables classifier-free guidance ON THE RENDER
    CONDITION:
        pred_final = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
    where pred_cond uses the supplied render_latents and pred_uncond uses
    render_latents=None. cfg_scale=1.0 is identical to no CFG (single
    forward pass per step, fastest). cfg_scale=3-7 amplifies conditioning
    sharply at the cost of 2× per-step compute.

    No-op when render_latents is None (no conditioning to amplify).
    """
    cond = cached["condition"].to(device=device, dtype=dtype, non_blocking=True)
    prompt = cached["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
    image  = cached["image_embeds"].to(device=device, dtype=dtype, non_blocking=True)
    rl = render_latents.to(device=device, dtype=dtype, non_blocking=True) if render_latents is not None else None
    use_cfg = (cfg_scale > 1.0) and (rl is not None)
    # v2 path: per-frame actions paired with the render condition. Drop them
    # in the unconditional branch (mirrors render_latents=None) so CFG is on
    # the *full embodiment conditioning*, not just render.
    actions = (
        cached["actions"].to(device=device, dtype=dtype, non_blocking=True).unsqueeze(0)
        if "actions" in cached else None
    )
    actions_cond = actions if rl is not None else None

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = torch.randn(
        clean_latents_for_first_frame.shape, generator=g, dtype=torch.float32
    ).to(device=device, dtype=dtype)

    # Preserve the first latent frame from the actual clean (Wan I2V convention).
    # First-frame latent comes from VAE-encoding the conditioning image. Here
    # we re-use the cached clean's first frame.
    first = clean_latents_for_first_frame[:, :, 0:1].to(device=device, dtype=dtype)

    scheduler.set_timesteps(num_inference_steps, device=device)
    for t in scheduler.timesteps:
        latents[:, :, 0:1] = first
        latent_in = torch.cat([latents, cond], dim=1)
        t_batch = t.expand(latents.shape[0]).to(device=device).float()
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            kw_cond = dict(
                hidden_states=latent_in,
                timestep=t_batch,
                encoder_hidden_states=prompt,
                encoder_hidden_states_image=image,
                render_latents=rl,
                return_dict=False,
            )
            if actions_cond is not None:
                kw_cond["actions"] = actions_cond
            pred_cond = model(**kw_cond)[0]
            if use_cfg:
                # Unconditional branch: drop both render and actions.
                pred_uncond = model(
                    hidden_states=latent_in,
                    timestep=t_batch,
                    encoder_hidden_states=prompt,
                    encoder_hidden_states_image=image,
                    render_latents=None,
                    return_dict=False,
                )[0]
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_cond
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]

    latents[:, :, 0:1] = first
    return latents


@torch.no_grad()
def _vae_decode(vae: WanVAEChunkedEncode, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of _encode_video_normalized: undo z-score + decode → (B,C,T,H,W) in [-1,1]."""
    latents = latents.to(vae.dtype)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    inv_std = (1.0 / torch.tensor(vae.config.latents_std)).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / inv_std + mean
    video = vae.decode(latents, return_dict=False)[0]   # (B, 3, T, H, W) in [-1, 1]
    return video


def _load_video_frames_uint8(
    mp4_path: str, num_frames: int, height: int, width: int,
) -> np.ndarray:
    """Decode an mp4, uniformly subsample to ``num_frames`` frames matching
    the SAME indexing the dataset uses (``np.linspace(0, T_src-1, num_frames)``),
    resize to (height, width). Returns (T, H, W, 3) uint8 RGB.

    This is the source-video equivalent of what the dataset/precompute does
    for the latents, so saving these alongside the model output makes them
    truly comparable.
    """
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    frames_bgr: list = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames_bgr.append(fr)
    cap.release()
    if not frames_bgr:
        raise RuntimeError(f"no frames decoded from {mp4_path}")
    T_src = len(frames_bgr)
    idxs = np.linspace(0, T_src - 1, num_frames).astype(np.int64)
    sel = [frames_bgr[int(i)] for i in idxs]
    out = np.empty((num_frames, height, width, 3), dtype=np.uint8)
    for k, f in enumerate(sel):
        if (f.shape[0], f.shape[1]) != (height, width):
            f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
        out[k] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    return out


def _save_mp4_from_uint8(arr: np.ndarray, fps: float, out_path: Path) -> None:
    """Save (T, H, W, 3) uint8 RGB as H.264 mp4 (faststart)."""
    T, H, W, _ = arr.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", f"{fps:.2f}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE,
    )
    proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait()


def _save_mp4(video: torch.Tensor, fps: float, out_path: Path) -> None:
    """Save a (B=1, 3, T, H, W) video tensor in [-1,1] as an H.264 mp4 (faststart)."""
    v = video[0].clamp(-1, 1).float()                       # (3, T, H, W)
    v = (v + 1.0) * 127.5
    v = v.clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0)  # (T, H, W, 3)
    arr = v.cpu().numpy()
    T, H, W, _ = arr.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", f"{fps:.2f}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE,
    )
    proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval_csv", required=True)
    p.add_argument("--dataset_base_path", default=None,
                   help="Override JSON dataset_base_path (e.g. data_wan_eval).")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument(
        "--cfg_scale", type=float, default=1.0,
        help="Classifier-free guidance scale on the render condition. "
             "1.0 = no CFG (single forward pass, fastest). 3.0-7.0 amplifies "
             "conditioning at 2× per-step cost. Useful when ghost-arm artifacts "
             "appear because the model is averaging over too many plausible "
             "robot trajectories. Doesn't affect the no_render or wrong_render "
             "ablations (no conditioning to amplify).",
    )
    p.add_argument("--num_samples", type=int, default=3,
                   help="How many videos from the eval CSV to generate.")
    p.add_argument("--do_ablations", action="store_true",
                   help="Also generate no_render and wrong_render variants.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ignore_prompts", action="store_true", default=None)
    p.add_argument(
        "--bias_subtract_npz", default=None,
        help="Path to an .npz file containing key 'shared_dir' (shape (D,)) "
             "— typically generations/v2_81f_epoch_29_bias_probe.npz. When set, "
             "the v2 action_adapter.out_proj output gets `bias_scale * shared_dir` "
             "added to it at every forward pass (via a forward hook). Set "
             "bias_scale < 0 to remove the bias; >0 to amplify it.",
    )
    p.add_argument(
        "--bias_scale", type=float, default=0.0,
        help="Multiplier applied to shared_dir before adding to contribution. "
             "0 = no change (default), -1 = full subtraction, -0.5 = half "
             "subtract, +1 = double the bias.",
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
    base = args.dataset_base_path or cfg["dataset_base_path"]
    ignore_prompts = args.ignore_prompts if args.ignore_prompts is not None else bool(
        cfg.get("ignore_prompts", False)
    )

    # ---- 1. dataset + precompute embeddings (text/image/VAE encoders) ----
    print(f"[gen] dataset: {args.eval_csv}")
    eval_ds = RenderI2VMetadataDataset(
        base_path=base,
        metadata_csv=args.eval_csv,
        num_frames=cfg["num_frames"],
        height=cfg["height"],
        width=cfg["width"],
        repeat=1,
        # v2 path: the actions stream is needed by model.forward. The dataset
        # raises if the CSV's actions column is partially populated, so use a
        # CSV where every row has an actions path (e.g. train_metadata_test.csv).
        action_stats_path=cfg.get("action_stats_path"),
        action_field=cfg.get("action_field", "state"),
    )
    n = min(args.num_samples, len(eval_ds.rows))
    print(f"[gen] generating {n} videos out of {len(eval_ds.rows)} in CSV")

    print(f"[gen] precomputing embeds for {n} samples …")
    # Subset the dataset rows so precompute only runs on what we need
    eval_ds.rows = eval_ds.rows[:n]
    embed_cache, _z_dim, scheduler = _precompute_embeddings(
        model_path=cfg["model_path"],
        dataset=eval_ds,
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

    # ---- 2. build DiT, load checkpoint ----
    print("[gen] building DiT and loading subset checkpoint …")
    dit = _load_dit_only(
        cfg["model_path"], dtype,
        use_embodiment_adapter=bool(cfg.get("use_embodiment_adapter", False)),
        embodiment_kwargs=cfg.get("embodiment_kwargs"),
        legacy_render_variant=cfg.get("legacy_render_variant", "egowm"),
        v2_adapter_kwargs=cfg.get("v2_adapter_kwargs"),
        v3_adapter_kwargs=cfg.get("v3_adapter_kwargs"),
    )
    _materialize_meta_submodules(dit)
    if hasattr(dit, "reset_zero_gates"):
        dit.reset_zero_gates()
    _load_ckpt_subset(dit, args.ckpt)
    dit = dit.to(device=device, dtype=dtype).eval()

    # ---- 2a. (optional) bias-direction debug hook ────────────────────────
    # Adds `bias_scale * shared_dir` to action_adapter.out_proj's output on
    # every forward pass. shared_dir is a (D,) tensor measured offline from
    # the v2 action_adapter (see scripts/probe_action_adapter_bias.py).
    # Used to test whether the universal "tint" is caused by the shared
    # bias direction baked into out_proj. bias_scale = -1 fully removes
    # the bias direction; +1 doubles it. 0 is a no-op (default).
    if args.bias_subtract_npz is not None and abs(args.bias_scale) > 1e-9:
        import numpy as np
        blob = np.load(args.bias_subtract_npz)
        if "shared_dir" not in blob.files:
            raise KeyError(f"{args.bias_subtract_npz} missing 'shared_dir' key")
        shared_dir = torch.from_numpy(blob["shared_dir"]).to(
            device=device, dtype=dtype,
        )  # (D,)
        assert shared_dir.ndim == 1, f"shared_dir must be 1D, got {shared_dir.shape}"
        scale_t = torch.tensor(args.bias_scale, device=device, dtype=dtype)
        print(f"[gen] BIAS-HOOK installed on action_adapter.out_proj  "
              f"scale={args.bias_scale}  shared_dir.norm={shared_dir.float().norm().item():.4f}  "
              f"npz={args.bias_subtract_npz}")
        def _bias_hook(_module, _inp, out):
            # out: (B, num_tokens, D)  — broadcast add (D,) per channel
            return out + scale_t * shared_dir
        if not hasattr(dit, "action_adapter"):
            raise RuntimeError(
                "bias hook requires legacy_render_variant='v2' (no action_adapter found)"
            )
        dit.action_adapter.out_proj.register_forward_hook(_bias_hook)

    # ---- 3. load VAE for decoding (precompute deletes its instance) ----
    print("[gen] loading VAE for decode …")
    vae = WanVAEChunkedEncode.from_pretrained(
        cfg["model_path"], subfolder="vae", torch_dtype=dtype,
    ).to(device).eval()

    # ---- 4. generate ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gen] writing videos under {out_dir}")

    # Per-sample metrics get accumulated for an aggregate report at the end.
    all_metrics: list = []

    for i in range(n):
        row = eval_ds.rows[i]
        scene = Path(row["video"]).stem if "/" in row["video"] else row["video"]
        cached = embed_cache[i]
        clean = cached["clean_latents"].to(device=device, dtype=dtype)
        rl = cached["render_latents"].to(device=device, dtype=dtype)

        print(f"\n[gen] [{i + 1}/{n}] scene={scene}")

        # Decode ground truth ONCE (VAE-decoded reference: removes VAE
        # compression error from PSNR/SSIM, isolates model error).
        ref_video = _vae_decode(vae, clean)
        ref_path = out_dir / f"{scene}_ref_vae_decoded.mp4"
        _save_mp4(ref_video, fps=15.0, out_path=ref_path)
        print(f"  ✓ {ref_path}  (reference: VAE-decoded ground truth)")

        sample_metrics: dict = {"scene": scene}

        # variant A: correct render (CFG-amplified if cfg_scale > 1)
        latents = _denoise(dit, scheduler, clean, cached, rl,
                           args.num_inference_steps, device, dtype,
                           seed=args.seed + i, cfg_scale=args.cfg_scale)
        gen_a = _vae_decode(vae, latents)
        out = out_dir / f"{scene}_generated.mp4"
        _save_mp4(gen_a, fps=15.0, out_path=out)
        m_a = compute_psnr_ssim(gen_a, ref_video)
        sample_metrics["correct_render"] = m_a
        print(f"  ✓ {out}    PSNR={m_a['psnr_mean']:.2f} dB  SSIM={m_a['ssim_mean']:.4f}")

        if args.do_ablations:
            # B: no render conditioning at all (CFG inactive — nothing to amplify)
            latents = _denoise(dit, scheduler, clean, cached, None,
                               args.num_inference_steps, device, dtype,
                               seed=args.seed + i, cfg_scale=1.0)
            gen_b = _vae_decode(vae, latents)
            out_b = out_dir / f"{scene}_generated_no_render.mp4"
            _save_mp4(gen_b, fps=15.0, out_path=out_b)
            m_b = compute_psnr_ssim(gen_b, ref_video)
            sample_metrics["no_render"] = m_b
            print(f"  ✓ {out_b}  PSNR={m_b['psnr_mean']:.2f} dB  SSIM={m_b['ssim_mean']:.4f}")

            # C: WRONG render — pick a render that is genuinely different from
            #    the correct one. The cheap "next sample" trick fails with
            #    num_samples=1 (j == i → same render → identical generation).
            #    Two robust strategies, in order:
            #      (i)  if there's a different sample available, use its render
            #      (ii) otherwise, time-reverse the correct render's latent
            #           frames — same scene appearance but trajectories flipped.
            #           Still a "wrong" condition vs the actual target.
            if n >= 2:
                j = (i + 1) % n
                rl_wrong = embed_cache[j]["render_latents"].to(device=device, dtype=dtype)
                wrong_strategy = f"render of sample[{j}]"
            else:
                # Reverse temporal order on dim=2 (T_lat).
                rl_wrong = torch.flip(rl, dims=(2,)).contiguous()
                wrong_strategy = "time-reversed correct render"
            latents = _denoise(dit, scheduler, clean, cached, rl_wrong,
                               args.num_inference_steps, device, dtype,
                               seed=args.seed + i, cfg_scale=args.cfg_scale)
            gen_c = _vae_decode(vae, latents)
            out_c = out_dir / f"{scene}_generated_wrong_render.mp4"
            _save_mp4(gen_c, fps=15.0, out_path=out_c)
            m_c = compute_psnr_ssim(gen_c, ref_video)
            sample_metrics["wrong_render"] = m_c
            print(f"  ✓ {out_c}  PSNR={m_c['psnr_mean']:.2f} dB  SSIM={m_c['ssim_mean']:.4f}  ({wrong_strategy})")

        # Save real + render at the SAME number of frames + fps as the
        # generated/ref clips, so all four videos play at the same speed for
        # side-by-side comparison. Otherwise the source mp4s (60 fps × 745
        # frames ≈ 12 s) play 10× longer than the 17-frame model output.
        def _abs(p):
            return p if os.path.isabs(p) else os.path.join(base, p)
        try:
            real_subsampled = _load_video_frames_uint8(
                _abs(row["video"]), cfg["num_frames"], cfg["height"], cfg["width"],
            )
            _save_mp4_from_uint8(real_subsampled, fps=15.0,
                                 out_path=out_dir / f"{scene}_real.mp4")
        except Exception as e:
            print(f"  [WARN] could not subsample real mp4: {e}")
        try:
            render_subsampled = _load_video_frames_uint8(
                _abs(row["render"]), cfg["num_frames"], cfg["height"], cfg["width"],
            )
            _save_mp4_from_uint8(render_subsampled, fps=15.0,
                                 out_path=out_dir / f"{scene}_render.mp4")
        except Exception as e:
            print(f"  [WARN] could not subsample render mp4: {e}")

        all_metrics.append(sample_metrics)

    # ---- 5. aggregate metrics report ----
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS  (vs VAE-decoded ground truth)")
    print("=" * 80)

    def _mean(key, sub):
        vals = [m[sub][key] for m in all_metrics if sub in m]
        return float(np.mean(vals)) if vals else float("nan")

    variants = ["correct_render"]
    if args.do_ablations:
        variants += ["no_render", "wrong_render"]
    print(f"  {'variant':<18s}  {'PSNR (mean)':<12s}  {'SSIM (mean)':<12s}  n_samples")
    for v in variants:
        n_v = sum(1 for m in all_metrics if v in m)
        print(f"  {v:<18s}  {_mean('psnr_mean', v):<12.3f}  {_mean('ssim_mean', v):<12.4f}  {n_v}")

    # Save full metrics JSON next to the videos
    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": args.config,
            "ckpt": args.ckpt,
            "eval_csv": args.eval_csv,
            "num_inference_steps": args.num_inference_steps,
            "n_samples": n,
            "per_sample": all_metrics,
            "aggregate": {
                v: {
                    "psnr_mean": _mean("psnr_mean", v),
                    "psnr_min":  _mean("psnr_min",  v),
                    "ssim_mean": _mean("ssim_mean", v),
                    "ssim_min":  _mean("ssim_min",  v),
                }
                for v in variants
            },
        }, f, indent=2)
    print(f"\n[gen] metrics → {json_path}")
    print(f"[gen] videos  → {out_dir}/")
    print("[gen] DONE.")


if __name__ == "__main__":
    main()
