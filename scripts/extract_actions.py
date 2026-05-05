#!/usr/bin/env python3
"""
Extract per-scene action / state trajectories from DROID trajectory.h5 files.

For every ``scene_*/trajectory.h5`` under ``--data_root``, write:

    <out_dir>/actions/<scene>.npz

with these arrays (all float32, length T = trajectory length):

    state              (T, 8)  joint_position (7) + gripper_position (1)
    action             (T, 7)  target_cartesian_position (6) + target_gripper_position (1)
    cartesian_position (T, 6)  observation/robot_state/cartesian_position
    joint_velocity     (T, 7)  observation/robot_state/joint_velocities
    gripper_position   (T,)    observation/robot_state/gripper_position

Both ``state`` and ``action`` are saved so downstream training can pick the
representation it wants without re-extracting.

Optionally merges an ``actions`` column into an existing
``<out_dir>/train_metadata.csv`` (per-scene path, relative to out_dir).

Usage
-----
    python scripts/extract_actions.py \\
        --data_root data/droid_1k \\
        --out_dir   data_wan_1k \\
        --update_csv

Idempotent: skips scenes whose ``actions/<scene>.npz`` already exists
unless ``--force`` is passed.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _to_f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def extract_one(traj_path: Path, out_path: Path) -> Dict[str, int]:
    import h5py

    with h5py.File(traj_path, "r") as f:
        # State (observed at the camera frame) ─────────────────────────────
        joint_pos = _to_f32(f["observation/robot_state/joint_positions"][:])     # (T, 7)
        gripper   = _to_f32(f["observation/robot_state/gripper_position"][:])    # (T,)
        cart_obs  = _to_f32(f["observation/robot_state/cartesian_position"][:])  # (T, 6)
        joint_vel = _to_f32(f["observation/robot_state/joint_velocities"][:])    # (T, 7)

        # Action (commanded target) ────────────────────────────────────────
        tgt_cart  = _to_f32(f["action/target_cartesian_position"][:])            # (T, 6)
        tgt_grip  = _to_f32(f["action/target_gripper_position"][:])              # (T,)

    T = joint_pos.shape[0]
    if not (gripper.shape[0] == cart_obs.shape[0] == tgt_cart.shape[0] == tgt_grip.shape[0] == T):
        raise ValueError(
            f"{traj_path}: inconsistent T across streams "
            f"(joint={T}, grip={gripper.shape[0]}, cart={cart_obs.shape[0]}, "
            f"tgt_cart={tgt_cart.shape[0]}, tgt_grip={tgt_grip.shape[0]})"
        )

    state  = np.concatenate([joint_pos, gripper[:, None]], axis=1).astype(np.float32)  # (T, 8)
    action = np.concatenate([tgt_cart, tgt_grip[:, None]], axis=1).astype(np.float32)  # (T, 7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        state=state,
        action=action,
        cartesian_position=cart_obs,
        joint_velocity=joint_vel,
        gripper_position=gripper.astype(np.float32),
    )
    return {"T": T}


def discover_scenes(data_root: Path) -> List[Path]:
    return sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("scene_") and (d / "trajectory.h5").is_file()
    )


def update_metadata_csv(csv_path: Path, actions_dir_rel: str) -> int:
    """Add/update an `actions` column on train_metadata.csv.

    Matches rows on ``video`` basename → ``actions/<scene>.npz``.
    Returns the number of rows updated.
    """
    if not csv_path.is_file():
        logging.warning("[csv] %s not found – skipping CSV update", csv_path)
        return 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "actions" not in fieldnames:
        fieldnames.append("actions")

    out_root = csv_path.parent
    n_set = n_skipped = 0
    for row in rows:
        video = row.get("video", "")
        if not video:
            continue
        p = Path(video)
        scene = ""
        for anc in [p.parent, *p.parents]:
            if anc.name.startswith("scene_"):
                scene = anc.name
                break
        if not scene:
            scene = p.stem if p.stem.startswith("scene_") else ""
        if not scene:
            continue
        rel = f"{actions_dir_rel}/{scene}.npz"
        # Only set the column if the file actually exists; rows whose action
        # extraction failed (e.g. missing action/target_cartesian_position in
        # an older lab format) keep an empty actions field.
        if (out_root / rel).is_file():
            row["actions"] = rel
            n_set += 1
        else:
            row["actions"] = ""
            n_skipped += 1
    if n_skipped:
        logging.info("[csv] %d row(s) had no .npz on disk; left actions empty", n_skipped)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return n_set


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, required=True,
                   help="Root containing scene_*/trajectory.h5 (e.g. data/droid_1k).")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Output root (e.g. data_wan_1k); writes <out_dir>/actions/<scene>.npz.")
    p.add_argument("--update_csv", action="store_true",
                   help="Add an `actions` column to <out_dir>/train_metadata.csv after extraction.")
    p.add_argument("--csv_name", type=str, default="train_metadata.csv")
    p.add_argument("--force", action="store_true", help="Overwrite existing per-scene .npz files.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[actions] %(levelname)s %(message)s")

    scenes = discover_scenes(args.data_root)
    if not scenes:
        logging.error("No scene_*/trajectory.h5 found under %s", args.data_root)
        sys.exit(1)

    actions_dir = args.out_dir / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_skip = n_fail = 0
    for sd in scenes:
        out_npz = actions_dir / f"{sd.name}.npz"
        if out_npz.is_file() and not args.force:
            n_skip += 1
            continue
        try:
            info = extract_one(sd / "trajectory.h5", out_npz)
            n_done += 1
            if n_done % 50 == 0:
                logging.info("  …%d/%d (last: %s, T=%d)", n_done, len(scenes), sd.name, info["T"])
        except Exception as exc:
            n_fail += 1
            logging.error("FAIL %s: %s", sd.name, exc)

    logging.info("done: extracted=%d skipped(existing)=%d failed=%d total=%d",
                 n_done, n_skip, n_fail, len(scenes))

    if args.update_csv:
        csv_path = args.out_dir / args.csv_name
        n_set = update_metadata_csv(csv_path, actions_dir_rel="actions")
        logging.info("CSV: set actions= for %d rows in %s", n_set, csv_path)


if __name__ == "__main__":
    main()
