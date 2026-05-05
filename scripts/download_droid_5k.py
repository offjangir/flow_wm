#!/usr/bin/env python3
"""
Fast public DROID downloader (no sudo, no HF token, no Google SDK).

Pulls the raw H5 + MP4 layout that scale_data_generation.py expects from the
anonymous public GCS bucket:

    https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/

Pipeline
--------
  fetch-meta   Download aggregated-annotations-030724.json (~12 MB) to a cache.
  sample       Stratified diverse sample of N episodes:
                 1. Build candidate set from annotations (50k UUIDs).
                 2. Compute each UUID's GCS bucket path.
                 3. Parallel HEAD-validate (drop episodes not in the bucket).
                 4. Stratify by task category x lab, with session cap by date.
                 -> writes manifest.csv
  download     Parallel HTTPS download of each manifest row:
                 trajectory.h5 + metadata_<UUID>.json + recordings/MP4/<serial>.mp4
                 Layout is exactly what scale_data_generation.py reads.
                 Idempotent: re-running skips already-complete scenes.
  extract      (Optional) Parse each scene's trajectory.h5 ->
                 joints.npy, intrinsics.npy, extrinsics.npy, prompt.txt

Output layout
-------------
  <out_dir>/manifest.csv
  <out_dir>/scene_<idx>/trajectory.h5
  <out_dir>/scene_<idx>/recordings/MP4/<serial>.mp4
  <out_dir>/scene_<idx>/metadata.json
  <out_dir>/scene_<idx>/{joints,intrinsics,extrinsics}.npy   # only after `extract`
  <out_dir>/scene_<idx>/prompt.txt                            # only after `extract`

Diversity axes (sample step)
----------------------------
  1. Task category   - 8 keyword-clustered groups (~target/8 each)
  2. Lab / building  - uniform across the 13 DROID labs
  3. Session cap     - max MAX_PER_SESSION episodes per (lab, recording date)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── Constants ────────────────────────────────────────────────────────────────

GCS_BASE = "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1"
ANNOTATIONS_URL = f"{GCS_BASE}/aggregated-annotations-030724.json"

# 8 task categories matched by simple keyword clustering on language instructions.
TASK_CATEGORIES: Dict[str, List[str]] = {
    "pick_place":   ["move object", "pick", "place", "grasp", "relocat",
                     "put object", "transfer", "lift", "drop"],
    "container":    ["container", "drawer", "bin", "basket", "box", "hamper",
                     "washer", "plate", "bowl", "tray", "shelf", "cup", "into the",
                     "out of the", "into a", "out of a"],
    "open_close":   ["open", "close", "hinged", "microwave", "oven", "book",
                     "dryer", "toilet", "cabinet", "fridge", "refrigerator", "lid"],
    "slide":        ["slide", "slidable", "sliding", "toaster", "dresser", "push", "pull"],
    "button_knob":  ["button", "switch", "press", "elevator", "knob", "dial",
                     "lever", "turn on", "turn off"],
    "hang_arrange": ["hang", "unhang", "hook", "arrange", "organize", "stack"],
    "deformable":   ["fold", "spread", "clump", "cloth", "towel", "cord",
                     "cable", "garment", "clothes", "pour"],
    "multi_step":   ["three tasks", "multiple", "consecutive", "sequence",
                     "and then", "after"],
}

MAX_PER_SESSION = 5     # max episodes from same (lab, recording date)
DEFAULT_WORKERS = 64    # threads for HEAD validation + download

# ─── UUID -> bucket path ──────────────────────────────────────────────────────

_TS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(\d{2})h-(\d{2})m-(\d{2})s$")

def uuid_to_bucket_path(uuid: str) -> Optional[str]:
    """
    UUID format:    <LAB>+<8hex>+YYYY-MM-DD-HHh-MMm-SSs
    Bucket layout:  <LAB>/success/YYYY-MM-DD/Day_Mon_<d>_HH:MM:SS_YYYY/

    Single-digit day uses double underscore (Oct__7), double-digit single (Apr_25).
    Returns None if UUID can't be parsed.
    """
    parts = uuid.split("+")
    if len(parts) != 3:
        return None
    lab, _hex, ts = parts
    m = _TS_RE.match(ts)
    if not m:
        return None
    Y, M, D, h, mi, s = (int(x) for x in m.groups())
    try:
        dt = datetime.datetime(Y, M, D, h, mi, s)
    except ValueError:
        return None
    day_abbr = dt.strftime("%a")          # e.g. "Sat"
    mon_abbr = dt.strftime("%b")          # e.g. "Oct"
    sep = "__" if D < 10 else "_"
    folder = f"{day_abbr}_{mon_abbr}{sep}{D}_{h:02d}:{mi:02d}:{s:02d}_{Y}"
    return f"{lab}/success/{Y:04d}-{M:02d}-{D:02d}/{folder}"


# ─── Task classifier ──────────────────────────────────────────────────────────

def classify_task(desc: str) -> str:
    low = (desc or "").lower()
    best_cat, best_hits = "other", 0
    for cat, kws in TASK_CATEGORIES.items():
        hits = sum(1 for kw in kws if kw in low)
        if hits > best_hits:
            best_hits, best_cat = hits, cat
    return best_cat


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def _make_session(workers: int):
    import requests
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    a = HTTPAdapter(pool_connections=workers, pool_maxsize=workers * 2,
                    max_retries=3)
    s.mount("https://", a)
    s.mount("http://",  a)
    return s


def _http_get_streaming(session, url: str, dest: Path, timeout: int = 120) -> Tuple[bool, Optional[str]]:
    """Stream URL to dest atomically. Returns (ok, error)."""
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.rename(dest)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _http_head_ok(session, url: str, timeout: int = 30) -> bool:
    try:
        r = session.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


# ─── Phase 1: fetch-meta ──────────────────────────────────────────────────────

def cmd_fetch_meta(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    out = cache / "aggregated-annotations-030724.json"
    if out.is_file() and out.stat().st_size > 0 and not args.force:
        print(f"[fetch-meta] cached: {out}  ({out.stat().st_size/1e6:.1f} MB)")
        return
    sess = _make_session(workers=4)
    print(f"[fetch-meta] {ANNOTATIONS_URL}")
    ok, err = _http_get_streaming(sess, ANNOTATIONS_URL, out)
    if not ok:
        sys.exit(f"[fetch-meta] FAILED: {err}")
    print(f"[fetch-meta] saved -> {out}  ({out.stat().st_size/1e6:.1f} MB)")


# ─── Phase 2: sample ──────────────────────────────────────────────────────────

def _load_candidates(cache: Path) -> List[Dict[str, Any]]:
    """
    Read annotations JSON and produce a list of candidate episode dicts.
    Schema in annotations:
       { uuid: {"language_instruction1": "...", "language_instruction2": "...", ...} }
    """
    path = cache / "aggregated-annotations-030724.json"
    if not path.is_file():
        sys.exit(f"Annotations not found at {path}. Run `fetch-meta` first.")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    cands: List[Dict[str, Any]] = []
    for uuid, val in raw.items():
        # Pick the first non-empty instruction
        instr = ""
        if isinstance(val, dict):
            for k in ("language_instruction1", "language_instruction2",
                     "language_instruction3", "language_instruction"):
                if k in val and val[k]:
                    instr = val[k]; break
        elif isinstance(val, str):
            instr = val
        elif isinstance(val, list) and val:
            instr = val[0]

        bucket_path = uuid_to_bucket_path(uuid)
        if bucket_path is None:
            continue

        lab = uuid.split("+")[0]
        date = uuid.split("+")[2][:10]    # YYYY-MM-DD

        cands.append({
            "uuid":         uuid,
            "bucket_path":  bucket_path,
            "lab":          lab,
            "date":         date,
            "task_desc":    instr,
            "task_cat":     classify_task(instr),
        })
    return cands


def _validate_in_bucket(
    cands: List[Dict[str, Any]],
    workers: int,
    sample_check: int,
) -> List[Dict[str, Any]]:
    """
    Parallel HEAD requests against <bucket_path>/trajectory.h5.
    Drops episodes not present in the bucket. If `sample_check` > 0 and < len(cands),
    only validates a random subset (faster, lower coverage).
    """
    sess = _make_session(workers)
    pool_targets = cands if sample_check <= 0 else random.sample(cands, min(sample_check, len(cands)))
    print(f"[sample] HEAD-validating {len(pool_targets):,} candidates with {workers} workers...")

    keep: List[Dict[str, Any]] = []
    n_done = n_ok = n_404 = 0
    t0 = time.time()
    futures = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ep in pool_targets:
            url = f"{GCS_BASE}/{ep['bucket_path']}/trajectory.h5"
            futures[pool.submit(_http_head_ok, sess, url)] = ep
        for fut in as_completed(futures):
            ep = futures[fut]
            ok = fut.result()
            n_done += 1
            if ok:
                n_ok += 1; keep.append(ep)
            else:
                n_404 += 1
            if n_done % 1000 == 0:
                el = time.time() - t0
                rate = n_done / el if el > 0 else 0
                eta = (len(pool_targets) - n_done) / rate if rate > 0 else 0
                print(f"  validated {n_done:,}/{len(pool_targets):,}  "
                      f"ok={n_ok:,} miss={n_404:,}  {rate:.0f}/s  ETA {eta:.0f}s",
                      flush=True)
    print(f"[sample] validation done. kept {len(keep):,}/{len(pool_targets):,}")
    return keep


def _stratified_sample(
    eps: List[Dict[str, Any]],
    n_target: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for ep in eps:
        by_cat[ep["task_cat"]].append(ep)
    cats = sorted(by_cat)
    per_cat = math.ceil(n_target / max(len(cats), 1))
    print(f"[sample] {len(cats)} task categories  ->  target {per_cat} per category")

    selected: List[Dict[str, Any]] = []
    for cat in cats:
        pool = by_cat[cat]
        rng.shuffle(pool)
        by_lab: Dict[str, List[Dict]] = defaultdict(list)
        for ep in pool: by_lab[ep["lab"]].append(ep)
        labs = sorted(by_lab)
        per_lab = math.ceil(per_cat / max(len(labs), 1))
        cat_sel: List[Dict[str, Any]] = []
        sess_count: Dict[str, int] = defaultdict(int)
        for lab in labs:
            picked = 0
            for ep in by_lab[lab]:
                if picked >= per_lab: break
                key = f"{ep['lab']}|{ep['date']}"
                if sess_count[key] >= MAX_PER_SESSION: continue
                cat_sel.append(ep); sess_count[key] += 1; picked += 1
        rng.shuffle(cat_sel)
        selected.extend(cat_sel[:per_cat])

    rng.shuffle(selected)
    selected = selected[:n_target]

    # Assign sortable scene IDs
    width = max(len(str(len(selected))), 5)
    for i, ep in enumerate(selected):
        ep["scene_id"] = f"{i:0{width}d}"
    return selected


def cmd_sample(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    cands = _load_candidates(cache)
    print(f"[sample] {len(cands):,} candidate episodes from annotations")

    # Pre-bucket-validate all (or a sub-sample for speed)
    valid = _validate_in_bucket(
        cands,
        workers=args.workers,
        sample_check=args.head_check_n,
    )
    if not valid:
        sys.exit("[sample] No episodes validated against bucket. Check connectivity.")

    selected = _stratified_sample(valid, n_target=args.n, seed=args.seed)

    fields = ["scene_id", "uuid", "bucket_path", "lab", "date", "task_cat", "task_desc"]
    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(selected)

    cat_ct: Dict[str, int] = defaultdict(int)
    labs: set = set()
    for ep in selected:
        cat_ct[ep["task_cat"]] += 1; labs.add(ep["lab"])
    print(f"\n[sample] selected {len(selected):,} across {len(labs)} labs")
    for cat in sorted(cat_ct):
        print(f"  {cat:15s}: {cat_ct[cat]}")
    print(f"\n[sample] manifest -> {manifest}")


# ─── Phase 3: download ────────────────────────────────────────────────────────

def _download_episode(
    ep: Dict[str, str],
    out_dir: Path,
    session,
    cameras: List[str],
) -> Dict[str, Any]:
    scene = out_dir / f"scene_{ep['scene_id']}"
    scene.mkdir(parents=True, exist_ok=True)
    base = f"{GCS_BASE}/{ep['bucket_path']}"
    errors: List[str] = []

    h5 = scene / "trajectory.h5"
    meta = scene / "metadata.json"

    # 1. trajectory.h5
    if not h5.is_file():
        ok, err = _http_get_streaming(session, f"{base}/trajectory.h5", h5)
        if not ok: errors.append(f"h5: {err}")

    # 2. metadata_<UUID>.json -> stored locally as metadata.json
    if not meta.is_file():
        ok, err = _http_get_streaming(
            session, f"{base}/metadata_{ep['uuid']}.json", meta)
        if not ok: errors.append(f"meta: {err}")

    # 3. parse metadata to find camera serials
    md: Dict[str, Any] = {}
    if meta.is_file() and meta.stat().st_size > 0:
        try:
            md = json.loads(meta.read_text())
        except Exception as e:
            errors.append(f"meta-parse: {e}")

    serial_keys = {
        "ext1":  "ext1_cam_serial",
        "ext2":  "ext2_cam_serial",
        "wrist": "wrist_cam_serial",
    }
    mp4_dir = scene / "recordings" / "MP4"
    mp4_dir.mkdir(parents=True, exist_ok=True)
    n_mp4 = 0
    for cam in cameras:
        serial = str(md.get(serial_keys.get(cam, ""), "") or "")
        if not serial:
            continue
        mp4 = mp4_dir / f"{serial}.mp4"
        if mp4.is_file() and mp4.stat().st_size > 0:
            n_mp4 += 1; continue
        ok, err = _http_get_streaming(
            session, f"{base}/recordings/MP4/{serial}.mp4", mp4)
        if ok: n_mp4 += 1
        else:  errors.append(f"mp4-{cam}: {err}")

    has_h5  = h5.is_file()  and h5.stat().st_size  > 0
    has_mp4 = n_mp4 > 0
    return {
        "scene_id": ep["scene_id"],
        "uuid":     ep["uuid"],
        "ok":       has_h5 and has_mp4 and not errors,
        "n_mp4":    n_mp4,
        "errors":   errors,
    }


def cmd_download(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.is_file():
        sys.exit(f"Manifest not found: {manifest}")

    with open(manifest, encoding="utf-8") as f:
        eps = list(csv.DictReader(f))
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    print(f"[download] {len(eps)} episodes -> {out_dir}")
    print(f"[download] cameras={cameras}  workers={args.workers}")

    sess = _make_session(workers=args.workers)
    n_ok = n_fail = 0
    fails: List[Dict[str, Any]] = []
    t0 = time.time()
    bytes0 = _total_bytes(out_dir)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_download_episode, ep, out_dir, sess, cameras): ep
                for ep in eps}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["ok"]: n_ok += 1
            else: n_fail += 1; fails.append(r)
            if i % 25 == 0 or i == len(eps):
                el = time.time() - t0
                rate = i / el if el > 0 else 0
                eta = (len(eps) - i) / rate if rate > 0 else 0
                MB = (_total_bytes(out_dir) - bytes0) / 1e6
                print(f"  {i:5d}/{len(eps)}  ok={n_ok} fail={n_fail}  "
                      f"{rate:.1f} ep/s  {MB/el:.1f} MB/s  ETA {eta/60:.1f} min",
                      flush=True)

    summary = out_dir / "download_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["scene_id", "uuid", "ok", "n_mp4", "errors"],
            extrasaction="ignore",
        )
        w.writeheader()
        for r in fails:
            w.writerow({**r, "errors": "; ".join(r["errors"])})

    print(f"\n[download] done. ok={n_ok} fail={n_fail}")
    if n_fail:
        print(f"[download] failures -> {summary} (re-run with same args to retry)")


def _total_bytes(p: Path) -> int:
    if not p.is_dir(): return 0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try: total += f.stat().st_size
            except OSError: pass
    return total


# ─── Phase 4: extract ─────────────────────────────────────────────────────────

def cmd_extract(args: argparse.Namespace) -> None:
    """
    Pull joints/intrinsics/extrinsics from each scene's trajectory.h5 into .npy.
    Useful for inspection; not required by scale_data_generation.py (which reads
    the H5 directly). Requires h5py + numpy.
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        sys.exit("extract requires h5py + numpy. Install: pip install h5py numpy")

    out_dir = Path(args.out_dir)
    scenes = sorted(out_dir.glob("scene_*"))
    print(f"[extract] {len(scenes)} scenes in {out_dir}")
    n_ok = n_fail = 0

    for scene in scenes:
        h5_path = scene / "trajectory.h5"
        meta_path = scene / "metadata.json"
        if not (h5_path.is_file() and meta_path.is_file()):
            n_fail += 1; continue

        try:
            md = json.loads(meta_path.read_text())
            serial = md.get(f"{args.camera_role}_cam_serial", "")
            cam_key = f"{serial}_left" if serial else None

            with h5py.File(h5_path, "r") as f:
                obs = f["observation"]

                # --- joint angles (T, 7) ---
                if "robot_state/joint_position" in obs:
                    np.save(scene / "joints.npy", obs["robot_state/joint_position"][...])
                elif "robot_state/joint_positions" in obs:
                    np.save(scene / "joints.npy", obs["robot_state/joint_positions"][...])

                # --- gripper (T,) ---
                for k in ("robot_state/gripper_position", "robot_state/gripper_positions"):
                    if k in obs:
                        np.save(scene / "grippers.npy", obs[k][...]); break

                # --- camera params for chosen role ---
                if cam_key and "camera_intrinsics" in obs and cam_key in obs["camera_intrinsics"]:
                    np.save(scene / "intrinsics.npy", obs["camera_intrinsics"][cam_key][...])
                if cam_key and "camera_extrinsics" in obs and cam_key in obs["camera_extrinsics"]:
                    np.save(scene / "extrinsics.npy", obs["camera_extrinsics"][cam_key][...])

            # --- prompt ---
            prompt = (md.get("current_task")
                      or md.get("language_instruction")
                      or md.get("language_instruction1") or "")
            (scene / "prompt.txt").write_text(prompt + "\n")

            n_ok += 1
            if n_ok % 200 == 0: print(f"  ok={n_ok}", flush=True)
        except Exception as e:
            print(f"  {scene.name}: {e}"); n_fail += 1

    print(f"[extract] done. ok={n_ok} fail={n_fail}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("fetch-meta", help="Download annotations JSON to cache")
    sp.add_argument("--cache_dir", required=True, type=str)
    sp.add_argument("--force", action="store_true", help="Re-download even if cached")

    sp = sub.add_parser("sample", help="Stratified sample -> manifest.csv")
    sp.add_argument("--cache_dir", required=True, type=str)
    sp.add_argument("--out_dir",   required=True, type=str,
                    help="Where to write manifest.csv (and later: scene dirs)")
    sp.add_argument("--n", type=int, default=5000, help="Target episodes (default 5000)")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    sp.add_argument("--head_check_n", type=int, default=0,
                    help="If >0, validate only this many random candidates (faster).")

    sp = sub.add_parser("download", help="Parallel download of selected episodes")
    sp.add_argument("--manifest", required=True, type=str)
    sp.add_argument("--out_dir",  required=True, type=str)
    sp.add_argument("--workers",  type=int, default=DEFAULT_WORKERS)
    sp.add_argument("--cameras",  default="ext1",
                    help="Comma-separated cameras to fetch: ext1,ext2,wrist (default ext1)")

    sp = sub.add_parser("extract", help="Extract npy/.txt artifacts from each scene")
    sp.add_argument("--out_dir",     required=True, type=str)
    sp.add_argument("--camera_role", default="ext1", choices=["ext1", "ext2", "wrist"])

    args = p.parse_args()
    {
        "fetch-meta": cmd_fetch_meta,
        "sample":     cmd_sample,
        "download":   cmd_download,
        "extract":    cmd_extract,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
