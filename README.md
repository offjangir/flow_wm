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

## Setup on another machine (start training elsewhere)

To clone this repo on a new node and start a v2 (action-conditioned) FSDP run, you need three things in addition to the code: (a) the `dr` conda env, (b) the training data under `data/droid_1k/` + `data_wan_1k/`, and (c) the eval data under `data_wan_eval/`. Pick the data path that fits your situation.

### 1. Clone + env

```bash
git clone git@github.com:offjangir/flow_wm.git
cd flow_wm
bash scripts/setup_dr_env_and_checkpoints.sh   # creates 'dr' conda env, downloads HF Wan 2.1 ckpt
conda activate dr
export PYTHONPATH=src
```

### 2. Data

There are two ways to get the data on the new machine. Pick one.

**Option A — rsync from a machine that already has it (fast).** If both machines see the same shared filesystem (`weka`), just point at the existing path. Otherwise rsync the processed data; raw DROID is the bulky part:

```bash
# On the source machine (e.g. c001), rsync the processed data:
rsync -avh --progress \
    /weka/scratch/hbharad2/users/yjangir1/flow_wm/data_wan_1k/ \
    user@new-host:/path/to/flow_wm/data_wan_1k/

# Eval data (only ~2 GB):
rsync -avh --progress \
    /weka/scratch/hbharad2/users/yjangir1/flow_wm/data_wan_eval/ \
    user@new-host:/path/to/flow_wm/data_wan_eval/

# Raw DROID (only needed if you'll re-extract actions or run the pipeline; ~8 GB):
rsync -avh --progress \
    /weka/scratch/hbharad2/users/yjangir1/flow_wm/data/droid_1k/ \
    user@new-host:/path/to/flow_wm/data/droid_1k/
```

Approximate sizes:

| Path | Size | Needed for | Notes |
|------|------|-----------|-------|
| `data/droid_1k/` | ~8 GB | re-extracting actions / rebuilding pipeline | trajectory.h5 + raw MP4s |
| `data_wan_1k/videos,clips,renders` | ~12 GB | training | symlinks + DrRobot renders |
| `data_wan_1k/alltracker_tracks/` | ~900 MB | training (tracks loss) | sparse tracks .npz |
| `data_wan_1k/alltracker_dense_tracks/` | ~185 GB | currently unused at training time | dense tracks .npz; can skip |
| `data_wan_1k/actions/` | ~30 MB | v2 action conditioning | created by `scripts/extract_actions.py` |
| `data_wan_1k/action_stats.json` | <1 KB | v2 z-score normalization | regeneratable |
| `data_wan_1k/train_metadata*.csv` | <1 MB | training | regeneratable from `prepare_data_wan.py` |
| `data_wan_eval/` | ~2 GB | inference / eval | held-out 10 scenes |

**Skip `alltracker_dense_tracks/`** unless you've added a training path that uses dense tracks — it's 185 GB and the current code only reads sparse tracks.

**Option B — rebuild the data pipeline on the new machine (slow).** Re-runs DROID episode sampling → DrRobot rendering → AllTracker tracks → action extraction. End-to-end this takes several hours per 1000 scenes:

```bash
# (i) Sample + download DROID episodes (default 5000; set N_EPISODES=1000 to match this dataset)
N_EPISODES=1000 OUT_DIR=data/droid_1k RUN_EXTRACT=1 RUN_RENDER=1 \
    bash scripts/run_droid_5k_pipeline.sh

# (ii) Build held-out eval set
bash scripts/build_eval_set.sh

# (iii) Extract 8-d action streams from trajectory.h5 → data_wan_1k/actions/*.npz
python scripts/extract_actions.py \
    --data_root data/droid_1k \
    --out_dir   data_wan_1k \
    --update_csv \
    --csv_name  train_metadata.csv

# (iv) Compute z-score stats over the train split
python -c "
import pandas as pd, numpy as np, json
df = pd.read_csv('data_wan_1k/train_metadata_train.csv')
states = []
for rel in df['actions']:
    if rel:
        states.append(np.load(f'data_wan_1k/{rel}')['state'])
cat = np.concatenate(states, axis=0)
mean = cat.mean(0).astype(np.float32); std = np.maximum(cat.std(0), 1e-6).astype(np.float32)
json.dump({
    'fields': ['joint_0','joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','gripper'],
    'mean': mean.tolist(), 'std': std.tolist(),
    'source_csv': 'train_metadata_train.csv', 'n_rows': len(df), 'n_samples': int(cat.shape[0]),
    'action_dim': 8, 'representation': 'joint_position(7) + gripper_position(1)',
}, open('data_wan_1k/action_stats.json','w'), indent=2)
"
```

After either path, the action npz layout is:
- `data_wan_1k/actions/scene_<id>.npz` with fields: `state` (T, 8), `action` (T, 7) [optional], `cartesian_position`, `joint_velocity`, `gripper_position`
- `data_wan_1k/action_stats.json` — per-dim mean/std for z-score normalization

### 3. Sanity-check before launching

```bash
# Verifies the v2 init pipeline (zero-init out_proj, alive cross_attn.in_proj, identity-at-init forward).
python scripts/audit_v2_init.py
```

### 4. Launch training

For 81-frame action-conditioned v2:

```bash
# Smoke (10 videos, 2 epochs, ~10 min):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
    --config_file configs/accelerate_fsdp.yaml --num_processes 6 \
    -m world_model.wan_flow.train_fsdp \
    --config configs/train_drrobot_1k_10vids_v2_fsdp.json

# Full 30-epoch run (~50 hr on 6×A100-80GB):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
    --config_file configs/accelerate_fsdp.yaml --num_processes 6 \
    -m world_model.wan_flow.train_fsdp \
    --config configs/train_drrobot_1k_legacy_8xl40_action_aware_v2.json
```

The smoke must pass cleanly before committing to the full run.

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
