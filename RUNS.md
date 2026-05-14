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

**To eval this run:** add a `--legacy_render_variant` (or `--v3`) flag to
`eval_world_model.py` so `_build_pipeline` builds
`WanTransformerRenderConditionedV3` with `v3_adapter_kwargs`. Then the eval
command mirrors egowm's, plus:

```bash
    --action_stats_path data_wan_1k/action_stats.json \
    --action_field state \
    --legacy_render_variant v3 \
```

---

## Known eval-tooling gaps (TODO)

`eval_world_model.py` currently only fully supports `egowm` / `v1` / `v2`
(render-only) checkpoints. To close the loop on runs 2 and 3:

1. `--transformer_path` override — load a fine-tuned full transformer dir
   (unblocks full_dit / nordr eval).
2. v3 transformer-class selection in `_build_pipeline` — build
   `WanTransformerRenderConditionedV3` when evaluating a v3 checkpoint.
