#!/usr/bin/env python3
"""
Sample N held-out DROID episodes (NOT in an existing manifest).

Reads the same DROID annotations cache used by ``download_droid_5k.py``,
filters out any UUIDs present in ``--exclude_manifest``, and picks N at
random (seeded). Writes an output manifest in the same schema so the
existing ``download_droid_5k.py download`` subcommand can consume it.

Usage::

  python scripts/sample_eval_episodes.py \\
      --cache_dir data/droid_meta_cache \\
      --exclude_manifest data/droid_1k/manifest.csv \\
      --n 10 \\
      --seed 99 \\
      --out_manifest data/droid_eval/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Reuse the loader so we get the exact same record schema.
from download_droid_5k import _load_candidates, _validate_in_bucket  # type: ignore # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True, type=str)
    p.add_argument("--exclude_manifest", required=True, type=str,
                   help="CSV whose 'uuid' column lists episodes to exclude.")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--head_check_n", type=int, default=0)
    p.add_argument("--out_manifest", required=True, type=str)
    p.add_argument("--scene_id_offset", type=int, default=10000,
                   help="Start scene_ids at this number to avoid collision "
                        "with the existing manifest's IDs.")
    args = p.parse_args()

    # Build exclusion set
    excluded = set()
    with open(args.exclude_manifest, "r") as f:
        for r in csv.DictReader(f):
            excluded.add(r["uuid"])
    print(f"[sample-eval] excluding {len(excluded):,} UUIDs already in "
          f"{args.exclude_manifest}")

    # Load candidates (same path as download_droid_5k.py)
    cands = _load_candidates(Path(args.cache_dir))
    print(f"[sample-eval] {len(cands):,} candidate episodes")

    # Filter
    pool = [ep for ep in cands if ep["uuid"] not in excluded]
    print(f"[sample-eval] {len(pool):,} candidates after exclusion")

    # HEAD-validate (so we don't pick UUIDs that fail at download time)
    valid = _validate_in_bucket(
        pool, workers=args.workers, sample_check=args.head_check_n,
    )
    if not valid:
        sys.exit("[sample-eval] no candidates passed HEAD validation")

    # Random sample (no stratification — eval set is small, just want diverse)
    rng = random.Random(args.seed)
    rng.shuffle(valid)
    selected = valid[: args.n]
    print(f"[sample-eval] selected {len(selected)} episodes (seed={args.seed})")

    # Assign scene IDs starting from offset (won't collide with droid_1k IDs)
    width = max(5, len(str(args.scene_id_offset + len(selected))))
    for i, ep in enumerate(selected):
        ep["scene_id"] = f"{args.scene_id_offset + i:0{width}d}"

    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scene_id", "uuid", "bucket_path", "lab", "date", "task_cat", "task_desc"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(selected)
    print(f"[sample-eval] wrote {out}")
    for ep in selected:
        print(f"  scene_{ep['scene_id']} | lab={ep['lab']} | {ep['task_desc'][:60]}")


if __name__ == "__main__":
    main()
