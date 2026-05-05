# `download_droid_5k.py` — Sample & Download Diverse DROID Episodes

Fast, no-auth-required pipeline for grabbing a stratified sample of DROID
trajectories from the public Google Cloud Storage bucket and laying them
out in the directory format that `scripts/scale_data_generation.py`
already understands.

```
fetch-meta  →  sample  →  download  →  (extract)  →  scale_data_generation.py
```

No HuggingFace token. No `gcloud`/`aria2c` install. No sudo.
Pure Python over HTTPS to a public bucket.

---

## Why this exists

DROID is 76 000 trajectories / ~1.8 TB. Training the world model with
20 videos is a memorization regime; 5 000 videos is the smallest size
where the model has to actually *learn* motion priors instead of overfit
to a single tabletop. This script pulls a **diverse** 5 k subset
(stratified by task category × lab, with a session cap by recording
date) so the sample isn't dominated by "move object" demos from one
afternoon at one institution.

---

## Prerequisites

- Python 3 with `requests` (already in the base conda env).
- For the optional `extract` phase: `h5py + numpy` — the `dr` conda env
  on this machine already has them.

That's it. No HF token, no Google SDK, nothing else to install.

---

## Pipeline

### 1. `fetch-meta` — download annotations cache (~12 MB, ~1 s)

```bash
python scripts/download_droid_5k.py fetch-meta \
    --cache_dir data/droid_meta_cache
```

Pulls `aggregated-annotations-030724.json` (50 092 episode UUIDs with
language instructions) from the GCS bucket. Cached so subsequent
`sample` runs reuse it.

### 2. `sample` — stratified diverse selection (~1–2 min for n=5000)

```bash
python scripts/download_droid_5k.py sample \
    --cache_dir data/droid_meta_cache \
    --out_dir   data/droid_5k \
    --n         5000 \
    --workers   64
```

Steps:
1. Reads annotations → 50 k candidate UUIDs.
2. Computes each UUID's GCS bucket path (`<LAB>/success/<date>/<ts>/`).
3. Parallel HEAD-validates against the bucket (~80 s with 64 workers).
   ~80 % of UUIDs are present in DROID 1.0.1; the rest are dropped.
4. Stratified sample across:
   - **8 task categories** (keyword-clustered from the language
     instruction — see `TASK_CATEGORIES` in the script).
   - **13 labs** (uniform within each task category).
   - **Session cap**: max 5 episodes per `(lab, date)` to avoid
     correlated demos from the same recording session.
5. Writes `<out_dir>/manifest.csv`.

Useful flags:
- `--head_check_n 1000` — only validate a random subset (faster, lower
  recall; use to debug).
- `--seed N` — different stratified sample.

### 3. `download` — parallel HTTPS pull (~10–15 min for 5000 eps / ~28 GB)

```bash
python scripts/download_droid_5k.py download \
    --manifest data/droid_5k/manifest.csv \
    --out_dir  data/droid_5k \
    --workers  64 \
    --cameras  ext1
```

Per episode, fetches:
- `trajectory.h5` (~1.4 MB)
- `metadata_<UUID>.json` (~10 KB) — saved as `metadata.json`
- `recordings/MP4/<serial>.mp4` (~4 MB) for each camera in `--cameras`

`--cameras` accepts comma-separated values: `ext1`, `ext2`, `wrist`.
The existing pipeline only needs one (default `ext1`).

**Idempotent.** Re-running skips files that already exist with non-zero
size, so interrupted runs resume cleanly. Failed episodes are listed in
`<out_dir>/download_summary.csv`; re-run the same command to retry.

### 4. `extract` (optional) — pull `.npy` artefacts for inspection

```bash
conda activate dr   # h5py + numpy needed
python scripts/download_droid_5k.py extract \
    --out_dir     data/droid_5k \
    --camera_role ext1
```

For each `scene_*/`, parses `trajectory.h5` and writes:
- `joints.npy`        — `(T, 7)` Franka joint angles
- `grippers.npy`      — `(T,)` gripper state
- `intrinsics.npy`    — `(3, 3)` camera intrinsic
- `extrinsics.npy`    — `(T, 6)` per-frame extrinsics (translation + Euler)
- `prompt.txt`        — task description from metadata

You **do not need this step to run training**. `scale_data_generation.py`
reads `trajectory.h5` directly. `extract` is only for inspection /
sanity-checking the data.

### 5. Hand off to the existing render + tracks pipeline

```bash
export MUJOCO_GL=egl
python scripts/scale_data_generation.py \
    --data_root data/droid_5k \
    --out_dir   data_wan_5k \
    --gpus 0 1 2 3 --no_viz
```

This is the existing pipeline — unchanged. It walks `scene_*/`
directories, runs DrRobot render + AllTracker, and emits the
`metadata.csv` the world-model trainer consumes.

---

## Output layout

```
data/droid_5k/
├── manifest.csv                           # from `sample`
├── download_summary.csv                   # from `download` (only if any failed)
└── scene_<idx>/                           # idx zero-padded for sortable listing
    ├── trajectory.h5                      # raw DROID, joints/cam params/timestamps
    ├── metadata.json                      # original DROID episode metadata
    ├── recordings/MP4/<serial>.mp4        # one per --cameras
    │
    │  -- only after `extract` --
    ├── joints.npy
    ├── grippers.npy
    ├── intrinsics.npy
    ├── extrinsics.npy
    └── prompt.txt
```

This is the layout `scripts/scale_data_generation.py` already expects;
no conversion step is required between download and the existing
pipeline.

---

## Size & speed (measured on this cluster)

| Phase                       | Time           | Size      |
| --------------------------- | -------------- | --------- |
| `fetch-meta`                | ~1 s           | 12 MB     |
| `sample` (n=5000, full HEAD)| ~80 s          | trivial   |
| `download` (5000 × ext1)    | ~10–15 min     | **~28 GB** |
| `download` (5000 × ext1+ext2)| ~20–30 min    | ~55 GB    |
| `download` (5000 × all 3)   | ~30–45 min     | ~80 GB    |

Single-stream curl from this machine to GCS: 22 MB/s.
Aggregate with 16 workers (measured): 58 MB/s.
With 64 workers: expect 80–110 MB/s on a 1 Gbps link.

---

## Diversity breakdown (typical)

After `sample --n 5000`, expect roughly:

| Category      | Count | Why it matters |
|---------------|-------|----------------|
| pick_place    | ~700  | Bulk of DROID; fine-motion grasping |
| container     | ~700  | Drawer / bin / box opening |
| open_close    | ~700  | Hinged objects (microwave, fridge, doors) |
| slide         | ~600  | Slidable objects, push/pull |
| button_knob   | ~600  | Discrete-state interactions |
| hang_arrange  | ~500  | Hooks, stacking |
| deformable    | ~400  | Cloth, cord, pour |
| multi_step    | ~300  | "do A then B" composite tasks |
| other         | ~500  | Uncategorized — pure language not matching keywords |

Across **all 13 DROID labs**: AUTOLab, CLVR, GuptaLab, ILIAD, IPRL,
IRIS, PennPAL, RAD, RAIL, REAL, RPL, TRI, WEIRD.

---

## Troubleshooting

- **`HTTP 404` on many episodes during validation** → expected. ~20 %
  of annotation UUIDs reference episodes that aren't in the public
  1.0.1 bucket. The script drops them automatically.
- **`download_summary.csv` non-empty after a run** → transient HTTP
  errors. Re-run the same `download` command; it skips successful
  scenes and only retries the failures.
- **`HTTP 401`** on any request → you're hitting a HuggingFace URL by
  mistake. This script only uses `storage.googleapis.com` (anonymous);
  401 means a previous version of the script was used. Re-pull `main`.
- **`extract` complains about missing keys** → DROID H5 schema differs
  slightly across labs. The script falls back gracefully on most paths;
  inspect the H5 with the snippet below if needed.

```python
import h5py
f = h5py.File("scene_00000/trajectory.h5", "r")
def walk(g, d=0):
    for k in g.keys():
        v = g[k]
        print("  "*d + k, getattr(v, "shape", "/"))
        if not hasattr(v, "shape") and d < 3: walk(v, d+1)
walk(f)
```

---

## What this script does **not** do

- It does **not** run AllTracker or DrRobot rendering — that's
  `scale_data_generation.py`'s job.
- It does **not** download stereo MP4s, SVO files, or
  `trajectory_im128.h5` (the per-frame 128 px image dump). Those are
  large and unused by the existing pipeline.
- It does **not** re-encode video. MP4s are stored as-is from DROID.

---

## Going faster (optional, no sudo)

If 64-worker pure-Python isn't fast enough, install `aria2` via conda
(no sudo) and switch to it:

```bash
conda install -c conda-forge aria2
```

Then write a small adapter that emits `aria2c -i input.txt -j 32 -x 8 -s 8`.
Not worth the effort below ~50 GB; pure Python comfortably saturates a
1 Gbps link.
