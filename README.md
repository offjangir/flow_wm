# World Model — render-conditioned Wan I2V

Train and run **Wan 2.1 image-to-video** with **per-frame render-video conditioning** on the **Hugging Face Diffusers** stack. At each timestep, a DrRobot render of the robot executing the action is VAE-encoded to `(B, C, T_lat, H_lat, W_lat)`, compressed to per-latent-frame embeddings, and added to the Wan timestep embedding before the AdaLN modulation projection (EgoWM Eq. 5). The DiT remains compatible with standard Wan 2.1 blocks.

The target video the DiT denoises is the *real camera video* (e.g. DROID recording); the render latents provide the action signal.

This repo is a **focused training/inference package**. It complements [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) conceptually (flow-matching SFT, CSV metadata) but does **not** depend on DiffSynth’s native `WanModel` implementation.

## Install

```bash
cd wm
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Dependencies: PyTorch, Diffusers, Transformers, Accelerate, etc. (see `pyproject.toml`). Resolve any `transformers` / `huggingface-hub` version pins in your environment before training.

## Layout

| Path | Purpose |
|------|---------|
| `src/world_model/wan_flow/model.py` | VAE chunking, `RenderLatentEncoder`, `WanTransformerRenderConditioned`, pipeline, builders |
| `src/world_model/wan_flow/data.py` | CSV dataset (`video`, `prompt`, `render`) |
| `src/world_model/wan_flow/train.py` | Accelerate training loop |
| `configs/train.example.json` | Example hyperparameters (use with `--config`) |
| `examples/dataset/metadata.csv` | Column template |

The module directory is still named `wan_flow` for backwards compatibility; the optical-flow path has been replaced with render-video conditioning.

Console script after install: `world-model-train`.

## Data prep (DROID + DrRobot)

`prepare_data_wan.py` symlinks real DROID camera MP4s as training targets and DrRobot renders as per-frame conditioning, then writes `metadata.csv`:

```bash
python prepare_data_wan.py              # uses DROID ext1 as target
python prepare_data_wan.py --camera_key ext2
```

This produces:

- `data_wan/clips/*.mp4`    — real DROID camera videos (targets)
- `data_wan/renders/*.mp4`  — DrRobot rendered robot motion (conditioning)
- `data_wan/metadata.csv`   — columns `video,prompt,render`

## Training

**Metadata CSV** (relative paths resolve under `--dataset_base_path`):

- `video`  — `.mp4` path (target, real camera)
- `prompt` — caption
- `render` — `.mp4` path (DrRobot render, conditioning)

```bash
accelerate config   # once
python -m world_model.wan_flow.train \
  --model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  --dataset_base_path /path/to/data \
  --metadata_csv /path/to/data/metadata.csv \
  --output_dir ./checkpoints/wan_render \
  --trainable render_only
```

Optional JSON overrides:

```bash
python -m world_model.wan_flow.train --config configs/train.example.json
```

**Modes**

- `render_only` (default): trains `render_encoder` + `render_fuse` only — lowest VRAM.
- `full_dit`: trains the full transformer.

Checkpoints: under `--output_dir/epoch_*`. In `render_only` mode, only the render conditioner weights are saved as `render_conditioner.pt`.

## Inference

Load with `RenderConditionedWanI2VPipeline` and pass `render_video=<frames>` or `render_latents=(B,C,T_lat,H_lat,W_lat)`. See `world_model.wan_flow.model.build_render_conditioned_wan_i2v(..., return_pipeline=True)`.

## Legacy import

`share_wanarch.py` at the repo root re-exports the package when `src/` is on the path (editable install recommended instead).

## License

Apache-2.0 (aligned with Diffusers / common model weights terms; check your base checkpoint license separately).
# flow_wm
