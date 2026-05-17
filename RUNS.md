# Training runs — checkpoints, configs, and how to evaluate

Reference for the three active DROID/DrRobot world-model runs. `checkpoints/`
is gitignored, so the run directories below live only on the cluster
(`/work/nvme/becx/yjangir/flow_wm/`).

Common eval entry point: `scripts/eval_world_model.py`. It builds the Wan I2V
pipeline, loads a trained checkpoint, generates a rollout per sample, and writes
a side-by-side MP4 (GT | render | prediction) plus per-frame + aggregate
PSNR/SSIM/MSE into `eval_summary.json`.

All three runs were trained with `ignore_prompts: true` (the text encoder only
ever saw the empty string), so eval must pass `--force_empty_prompt` and
`--guidance_scale 1.0` (text CFG is a no-op / out-of-distribution otherwise).

---

## 1. egowm — render-conditioned baseline

The EgoWM-style render conditioning (`legacy_render_variant: "egowm"`): per-token
spatial render encoder added directly to per-token temb. This is the baseline
the v2/v3 action-conditioned variants are measured against.

| | |
|---|---|
| Run dir | `checkpoints/wan_render_drrobot_1k_legacy_8xl40_egowm/run_20260513_042830/` |
| Base config | `configs/train_drrobot_1k_legacy_8xl40_egowm.json` |
| Launch script | `scripts/slurm/train_drrobot_1k_legacy_8xl40_egowm_scratch200e_4gpu.slurm` |
| Slurm overrides | `--num_epochs 200 --save_every_n_epochs 25` |
| wandb run | `1k_legacy_8xl40_egowm_scratch_200e_4gpu` |
| trainable | `render_only` + `unfreeze_last_n_blocks: 8` |
| Checkpoints | `epoch_{24,49,74,99,124}/render_conditioner.pt` (~13 GB each) |
| Status | running (job 2278999) |

### Eval (works today)

```bash
python scripts/eval_world_model.py \
    --model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --checkpoint_path checkpoints/wan_render_drrobot_1k_legacy_8xl40_egowm/run_20260513_042830/epoch_124/render_conditioner.pt \
    --metadata_csv data_wan_eval/eval_metadata.csv \
    --dataset_base_path data_wan_eval \
    --output_dir eval_outputs/egowm_epoch124_heldout \
    --num_samples 10 \
    --num_frames 17 --height 240 --width 432 \
    --num_inference_steps 50 --guidance_scale 1.0 \
    --force_empty_prompt --mixed_precision bf16 --no-cpu_offload
```

Swap `--metadata_csv data_wan_1k/train_metadata_train.csv --dataset_base_path
data_wan_1k` for an in-distribution (memorization) check instead of held-out.

---

## 2. nordr — vanilla full-DiT overfit (no render conditioning)

A vanilla Wan 2.1 I2V fine-tune: `trainable: full_dit`,
`--drop_render_conditioning`, `--lambda_tracks 0`. No render adapter, no tracks
head — the "can the base model overfit all 802 videos at all" baseline.

| | |
|---|---|
| Run dir | `checkpoints/wan_render_drrobot_1k_norender_overfit/run_20260513_045329/` |
| Base config | `configs/train_drrobot_1k_legacy_8xl40_egowm.json` (heavily overridden) |
| Launch script | `scripts/slurm/train_drrobot_1k_norender_overfit_200e_4gpu.slurm` |
| Slurm overrides | `--trainable full_dit --drop_render_conditioning --lambda_tracks 0.0 --num_epochs 200 --save_every_n_epochs 100` |
| wandb run | `1k_norender_overfit_200e_4gpu` |
| Checkpoints | full transformer dir at `epoch_{99,199}/` (~28 GB each, `save_pretrained` format — NOT `render_conditioner.pt`) |
| Status | running (job 2279171); first save at epoch 99 |

### Eval (needs an eval-script change — see gap below)

Because `trainable: full_dit`, the checkpoint is a complete diffusers transformer
directory, not the small `render_conditioner.pt` subset. `eval_world_model.py`
currently builds the transformer from `--model_path` and only loads a
`render_conditioner.pt` via `--checkpoint_path` — it has no `--transformer_path`
override to point at a fine-tuned full transformer.

**To eval this run, one of:**
- Add a `--transformer_path` flag to `eval_world_model.py` that loads the
  `epoch_N/` transformer dir in place of the base one, then run with
  `--no_render_conditioner` (the model was trained drop-render).
- Or assemble a model dir: symlink `epoch_N/` as `transformer/` alongside the
  base Wan `vae/`, `scheduler/`, `image_encoder/`, `text_encoder/` and pass that
  as `--model_path` with `--no_render_conditioner`.

---

## 3. v3 — action-conditioned render adapter (current best design)

`legacy_render_variant: "v3"` — `ActionRenderCrossAdapterV3`
(`src/world_model/wan_flow/embodiment_adapter_v3.py`) wrapped by
`WanTransformerRenderConditionedV3` (`src/world_model/wan_flow/model_v3.py`,
a subclass — `model.py` untouched). 3-layer render encoder + 2-layer action
encoder, LayerNorm on both, 2 traditional pre-norm transformer blocks (residual
cross-attn + residual FFN), 2-layer projection added to temb_base. Fixes the v2
collapse/divergence; validated by a 10-vid × 10-epoch smoke (val loss
1.52 → 0.33 monotonic).

| | |
|---|---|
| Run dir | `checkpoints/wan_render_drrobot_1k_legacy_8xl40_action_aware_v3/` (created on first save) |
| Config | `configs/train_drrobot_1k_legacy_8xl40_action_aware_v3.json` |
| Launch script | `scripts/slurm/train_drrobot_1k_v3_action_aware_4gpu.slurm` |
| Smoke config | `configs/train_drrobot_1k_10vids_v3_fsdp.json` (10 vids × 10 epochs) |
| wandb run | `1k_legacy_8xl40_action_aware_v3` |
| trainable | `render_only` + `unfreeze_last_n_blocks: 8` |
| Checkpoints | `epoch_{9,19,29}/render_conditioner.pt` (saves the `action_adapter_v3.*` subset) |
| Status | queued (job 2282048) — no checkpoints yet |

v3 needs the action stream: `--action_stats_path data_wan_1k/action_stats.json
--action_field state` (8-d joint+gripper), and the metadata CSV must have an
`actions` column.

### Eval (needs an eval-script change — see gap below)

`eval_world_model.py` `_build_pipeline` always builds
`WanTransformerRenderConditioned`. A v3 checkpoint requires
`WanTransformerRenderConditionedV3` instead. The pipeline + script already
accept and forward `actions` (`--action_stats_path`, `--action_field` exist),
but the transformer-class selection is not wired.

**To eval this run** (now wired — `eval_world_model.py` accepts
`--legacy_render_variant {v3,v4}` plus `--v{3,4}_adapter_kwargs`):

```bash
    --action_stats_path data_wan_1k/action_stats.json \
    --action_field state \
    --legacy_render_variant v3 \
    --v3_adapter_kwargs '{"action_dim": 8, "hidden_dim": 512, "num_heads": 8, "num_blocks": 2, "spatial_downsample": 2}'
```

---

## 4. v4 — render self-attn + cross-attn (targets render-encoder collapse)

`legacy_render_variant: "v4"` — `ActionRenderSelfCrossAdapterV4`
(`src/world_model/wan_flow/embodiment_adapter_v3.py`) wrapped by
`WanTransformerRenderConditionedV4`
(`src/world_model/wan_flow/model_v4.py`, subclass — `model.py` and
`model_v3.py` untouched).

Built on top of v3 (Pathway A) with three architectural additions
targeting the universal-bias collapse mode diagnosed by
`scripts/diagnose_egowm_collapse.py` (legacy egowm: `R_ratio = 6.3`,
`contrib_cos = 0.96` — encoder emits a near-constant direction across
scenes):

1. **Pre-encoder mean subtraction on VAE latents** (per channel,
   spatial-temporal). For DrRobot renders (~95% black canvas), the
   spatial mean approximates `f_VAE(empty canvas)`; subtracting leaves
   only the per-scene arm-pose deviation as input to the conv encoder.
2. **Temporal mixing in the middle conv** (`(1,3,3)` → `(3,3,3)` kernel
   with manual replicate-pad in the temporal dim) so boundary frames
   don't see zero-padding.
3. **Self-attention transformer blocks** on per-token features, BEFORE
   the cross-attn with actions. 3D positional encoding is added on
   render tokens, 1D positional encoding on action tokens, so attention
   can reason about spatial/temporal alignment instead of treating
   tokens as unordered sets.

Validated by `scripts/diagnose_v4_collapse.py` on `epoch_9` of the
17-frame full run: `R_ratio = 2.67` (vs egowm's 6.3, v2's 15),
`contrib_cos = 0.87` (vs egowm 0.96), `to_temb` effective rank 1677/2048
(no architectural rank collapse). Action-shuffle delta `0.49` and
render-shuffle delta `0.07` confirm actions are the dominant
conditioning signal at this scale.

| | |
|---|---|
| Run dir | `checkpoints/wan_render_drrobot_1k_v4_full_17f_4xH100NVL_n06/` |
| Full config | `configs/train_drrobot_1k_v4_full_17f.json` (802 clips × 100 epochs, accum=2, lambda_tracks=50, max_query_points=640) |
| Smoke config | `configs/train_drrobot_1k_v4_smoke.json` (10 vids × 5 epochs, plumbing validation) |
| wandb run | `1k_v4_full_17f_accum2_4xH100NVL_n06` |
| trainable | `render_only` + `unfreeze_last_n_blocks: 8` |
| Checkpoints | `epoch_{9,19,29,...}/render_conditioner.pt` — saves the `action_adapter_v4.*` subset + unfrozen DiT blocks + tracks_head |
| Params | adapter ≈ 30 M (vs v3's ~9 M) |

v4 needs the action stream just like v3: `action_stats_path`,
`action_field: state` (8-d joint+gripper), and the metadata CSV must
have an `actions` column. **Important:** the original v4 config draft
did not set `action_stats_path` (caught after the first launch was 14
epochs deep with raw actions) — the committed config sets it so any
restart / new run gets proper z-score normalization. See
[`feedback_action_stats_path`](../memory/feedback_action_stats_path.md).

### Launch (single 4-GPU node, FSDP)

```bash
# from repo root, in `dr` conda env
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/path/to/huggingface \
nohup conda run -n dr accelerate launch \
  --num_processes 4 \
  --config_file configs/accelerate_fsdp.yaml \
  -m world_model.wan_flow.train_fsdp \
  --config configs/train_drrobot_1k_v4_full_17f.json \
  > logs/v4_full_17f_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Always pass `--config_file configs/accelerate_fsdp.yaml` — otherwise
accelerate silently falls back to DDP and only the `[FSDP train]`
log labels are correct, not the strategy
(see [`feedback_fsdp_config_file`](../memory/feedback_fsdp_config_file.md)).

### Smoke test (recommended before the full run)

```bash
accelerate launch --config_file configs/accelerate_fsdp.yaml \
  --num_processes 4 \
  -m world_model.wan_flow.train_fsdp \
  --config configs/train_drrobot_1k_v4_smoke.json
```

10 videos × 5 epochs (~10 min). Validates the v4 plumbing end-to-end:
adapter loads cleanly, FSDP shards correctly, forward + backward
complete, val computes, tracks loss contributes, checkpoint saves with
the `action_adapter_v4.` prefix.

### Eval (the same script supports v4 via `--legacy_render_variant v4`)

```bash
python scripts/eval_world_model.py \
  --model_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  --ckpt checkpoints/wan_render_drrobot_1k_v4_full_17f_*/run_*/epoch_9/render_conditioner.pt \
  --eval_csv data_wan_eval/eval_metadata.csv \
  --action_stats_path data_wan_1k/action_stats.json \
  --action_field state \
  --legacy_render_variant v4 \
  --v4_adapter_kwargs '{"action_dim": 8, "hidden_dim": 512, "num_heads": 8, "num_self_blocks": 2, "num_cross_blocks": 2, "spatial_downsample": 2, "max_action_frames": 128, "max_t": 32, "max_h": 32, "max_w": 64}'
```

The `v4_adapter_kwargs` MUST match the values used at training
(see the training config's `v4_adapter_kwargs` block).

### Diagnostic probe (collapse analysis)

```bash
PYTHONPATH=src python scripts/diagnose_v4_collapse.py \
  --ckpt checkpoints/wan_render_drrobot_1k_v4_full_17f_*/run_*/epoch_9/render_conditioner.pt \
  --eval-csv data_wan_1k/train_metadata_test.csv \
  --cache-path data_wan_1k/precompute_cache_probe_v4_16scenes_17f.pt \
  --n-scenes 16 \
  --out-dir diagnostics/v4_epoch9 \
  --device cuda:0 --dtype bf16
```

**Always use a probe-suffixed cache path** — the precompute helper
overwrites the cache file if the scene set differs from training's,
which silently clobbers the multi-GB train cache
(see [`feedback_probe_cache_clobber`](../memory/feedback_probe_cache_clobber.md)).

Writes `diagnostics/v4_epoch9/{metrics.json, diag.png, raw.npz}` with
the same metric names as `diagnose_egowm_collapse.py` and
`diagnose_v2_collapse.py` for direct cross-architecture comparison.

---

## Known eval-tooling gaps (TODO)

`eval_world_model.py` now supports `egowm` / `v1` / `v2` / `v3` / `v4`
checkpoints via `--legacy_render_variant`. Remaining gaps:

1. `--transformer_path` override — load a fine-tuned full transformer dir
   (unblocks full_dit / nordr eval).
