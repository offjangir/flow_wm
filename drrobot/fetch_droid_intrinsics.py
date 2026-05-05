#!/usr/bin/env python3
"""Fetch and cache ZED camera intrinsics for every scene in a DROID dataset.

For each scene_* folder, reads the metadata JSON to discover camera serial
numbers (wrist, ext1, ext2), then queries the Stereolabs public calibration
server to retrieve the factory intrinsics at the requested resolution.

Results are saved as a single JSON file (``droid_intrinsics_cache.json`` by
default) that can be consumed by ``render_droid_scenes.py`` or any other
downstream tool.

Usage:
    python fetch_droid_intrinsics.py \
        --dataset_root /data/user_data/yjangir/yash/sidegig/wm/data/droid_10_demos \
        --width 1280 --height 720
"""
import argparse
import configparser
import glob
import json
import os
import urllib.request
from typing import Dict, Optional

STEREOLABS_URL = "https://www.stereolabs.com/developers/calib/?SN={serial}"
CAMERA_KEYS = ("wrist", "ext1", "ext2")

RES_TO_SECTION = {
    (2208, 1242): "2K",
    (1920, 1080): "FHD",
    (1280, 720): "HD",
    (672, 376): "VGA",
}


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow 308 permanent redirects (Python <3.11 chokes on 308)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if code == 308:
            code = 301
        return super().redirect_request(req, fp, code, msg, headers, newurl)

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, code, msg, headers)


_url_opener = urllib.request.build_opener(_RedirectHandler)


def fetch_calibration_text(serial: str) -> str:
    url = STEREOLABS_URL.format(serial=serial)
    with _url_opener.open(url, timeout=15) as resp:
        return resp.read().decode("utf-8")


def parse_calibration(text: str) -> dict:
    cp = configparser.ConfigParser()
    cp.read_string(text)
    result = {}
    for section in cp.sections():
        d = {}
        for key in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"):
            if cp.has_option(section, key):
                d[key] = float(cp.get(section, key))
        if d:
            result[section] = d
    return result


def get_intrinsics(
    serial: str,
    width: int,
    height: int,
    eye: str,
    cache: Dict[str, dict],
) -> Optional[dict]:
    if serial not in cache:
        try:
            text = fetch_calibration_text(serial)
            cache[serial] = parse_calibration(text)
        except Exception as e:
            print(f"  WARNING: could not fetch calibration for {serial}: {e}")
            cache[serial] = {}

    all_sections = cache[serial]
    if not all_sections:
        return None

    res_tag = RES_TO_SECTION.get((width, height), "HD")
    side = "LEFT" if eye == "left" else "RIGHT"
    section = f"{side}_CAM_{res_tag}"
    if section not in all_sections:
        available = [s for s in all_sections if side in s]
        section = available[0] if available else None
    if section is None:
        return None
    return all_sections[section]


def main():
    parser = argparse.ArgumentParser(description="Fetch ZED intrinsics for DROID scenes")
    parser.add_argument(
        "--dataset_root", type=str,
        default="/data/user_data/yjangir/yash/sidegig/wm/data/droid_10_demos",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--eye", type=str, default="left", choices=["left", "right"])
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path. Default: <dataset_root>/droid_intrinsics_cache.json",
    )
    args = parser.parse_args()

    out_path = args.output or os.path.join(args.dataset_root, "droid_intrinsics_cache.json")

    scene_dirs = sorted(glob.glob(os.path.join(args.dataset_root, "scene_*")))
    if not scene_dirs:
        raise RuntimeError(f"No scene_* dirs in {args.dataset_root}")

    calib_cache: Dict[str, dict] = {}
    results = {}

    import math

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        meta_paths = glob.glob(os.path.join(scene_dir, "metadata_*.json"))
        if not meta_paths:
            print(f"[SKIP] {scene_name}: no metadata JSON")
            continue

        with open(meta_paths[0]) as f:
            metadata = json.load(f)

        scene_entry = {}
        for cam_key in CAMERA_KEYS:
            serial_field = f"{cam_key}_cam_serial"
            if serial_field not in metadata:
                print(f"  {cam_key}: serial not found in metadata")
                continue
            serial = str(metadata[serial_field])
            intrinsics = get_intrinsics(serial, args.width, args.height, args.eye, calib_cache)
            if intrinsics is None:
                print(f"  {cam_key} (serial={serial}): intrinsics not available")
                continue

            fx = intrinsics.get("fx", 0)
            fy = intrinsics.get("fy", 0)
            fov_x = math.degrees(2 * math.atan(args.width / (2 * fx))) if fx else None
            fov_y = math.degrees(2 * math.atan(args.height / (2 * fy))) if fy else None

            scene_entry[cam_key] = {
                "serial": serial,
                "resolution": [args.width, args.height],
                "eye": args.eye,
                **intrinsics,
                "fov_x_deg": round(fov_x, 2) if fov_x else None,
                "fov_y_deg": round(fov_y, 2) if fov_y else None,
            }
            print(
                f"  {cam_key} serial={serial}  fx={fx:.2f}  fy={fy:.2f}  "
                f"fov_x={fov_x:.1f}°  fov_y={fov_y:.1f}°"
            )

        if scene_entry:
            results[scene_name] = scene_entry

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved intrinsics for {len(results)} scenes to {out_path}")


if __name__ == "__main__":
    main()
