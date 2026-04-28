#!/usr/bin/env python3
"""
Batch pipeline: DROID extracted scenes -> DrRobot render -> Any4D flow -> WM training data.

For each scene with valid data under droid_extract_whole:
  1. Renders the joint trajectory through a trained DrRobot Gaussian model
  2. Computes dense 2D optical flow (via Any4D 3D scene flow, or OpenCV fallback)
  3. Assembles the original DROID camera images into a video
  4. Extracts the task prompt from DROID metadata
  5. Writes a metadata.csv ready for world-model training

Each scene directory is expected to contain:
  joints.npy       (T, 7) Franka joint angles in radians
  extrinsics0.npy  (4, 4) world-to-camera matrix (static camera)
  intrinsics.npy   (3, 3) camera intrinsic matrix
  grippers.npy     (T,)   gripper state (optional, unused)
  images0/         original DROID RGB frames
  metadata_*.json  DROID episode metadata (current_task, etc.)

Outputs under <output_root>/:
  metadata.csv                WM training CSV (video, prompt, flow columns)
  scene_*/
    drrobot_render.mp4        rendered robot video from DrRobot
    frames/frame_*.jpg        individual DrRobot rendered frames
    ref_binary_mask.png       foreground mask
    flow.npy                  (T, H, W, 2) float32 optical flow
    droid_video.mp4           original DROID camera footage
    prompt.txt                task description

Example:
  export MUJOCO_GL=egl
  python run_droid_scenes.py \\
    --dataset_root /data/group_data/katefgroup/datasets/droid_chenyu/droid_extract_whole \\
    --drrobot_model_path /data/user_data/yjangir/yash/sidegig/wm/drrobot/output/panda \\
    --output_root /data/user_data/yjangir/yash/sidegig/wm/droid_scene_outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DRROBOT_ROOT = SCRIPT_DIR / "drrobot"
ANY4D_ROOT = SCRIPT_DIR / "Any4D"
ANY4D_CHECKPOINT = ANY4D_ROOT / "checkpoints" / "any4d_4v_combined.pth"
EXTRACT_FLOW_SCRIPT = SCRIPT_DIR / "extract_any4d_flow.py"
DATASET_ROOT = Path(
    "/data/group_data/katefgroup/datasets/droid_chenyu/droid_extract_whole"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_scenes(dataset_root: Path) -> list[Path]:
    """Return sorted scene_* directories that contain joints.npy."""
    scenes = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and p.name.startswith("scene_") and (p / "joints.npy").is_file():
            scenes.append(p)
    return scenes


def fov_from_intrinsics(K: np.ndarray, width: int, height: int) -> tuple[float, float]:
    fx, fy = K[0, 0], K[1, 1]
    return 2.0 * math.atan(width / (2.0 * fx)), 2.0 * math.atan(height / (2.0 * fy))


def normalize_joints(
    joint_rad: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    """Map real joint angles (radians) to [-1, 1] matching DrRobot's convention."""
    q = np.asarray(joint_rad, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower_limits, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper_limits, dtype=np.float64).reshape(-1)
    n_chain = len(lo)
    out = np.zeros(n_chain, dtype=np.float32)
    use = min(len(q), n_chain)
    if use == 0:
        return out
    scale = 2.0 / (hi[:use] - lo[:use] + 1e-8)
    out[:use] = np.clip((q[:use] - lo[:use]) * scale - 1.0, -1.0, 1.0).astype(
        np.float32
    )
    return out


def get_scene_image_size(scene_dir: Path) -> tuple[int, int]:
    """Return (width, height) from the first image in images0/."""
    img_dir = scene_dir / "images0"
    sample_imgs = sorted(img_dir.glob("*.jpg"))
    if sample_imgs:
        from PIL import Image

        with Image.open(sample_imgs[0]) as img:
            return img.size
    return (640, 360)


def get_droid_prompt(scene_dir: Path) -> str:
    """Extract task description from DROID metadata JSON."""
    for jf in scene_dir.glob("metadata_*.json"):
        with open(jf, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("current_task", "robot manipulation task")
    return "robot manipulation task"


# ---------------------------------------------------------------------------
# Step 1: DrRobot rendering
# ---------------------------------------------------------------------------


def render_scene_drrobot(
    scene_dir: Path,
    work_dir: Path,
    drrobot_model_path: str,
    fps: float,
    max_frames: int | None,
    frame_stride: int,
    image_width: int | None,
    image_height: int | None,
) -> tuple[Path, list[int]]:
    """Render a scene's joint trajectory through DrRobot.

    Returns (video_path, selected_indices).
    """
    import torch

    joints_all = np.load(scene_dir / "joints.npy")
    extrinsics = np.load(scene_dir / "extrinsics0.npy")
    intrinsics = np.load(scene_dir / "intrinsics.npy")

    if image_width is None or image_height is None:
        image_width, image_height = get_scene_image_size(scene_dir)

    fov_x, fov_y = fov_from_intrinsics(intrinsics, image_width, image_height)

    saved_argv = sys.argv[:]
    prev_cwd = os.getcwd()
    drrobot_root = DRROBOT_ROOT.resolve()
    os.chdir(drrobot_root)
    sys.path.insert(0, str(drrobot_root))

    try:
        sys.argv = [
            "droid_drrobot_render",
            "--model_path",
            str(drrobot_model_path),
        ]
        from gaussian_renderer import render
        from scene.cameras import Camera_Pose
        from video_api import initialize_gaussians

        gaussians, background_color, sample_cameras, _chain = initialize_gaussians()
        lower_limits, upper_limits = gaussians.chain.get_joint_limits()
        lower_limits = np.asarray(lower_limits, dtype=np.float64)
        upper_limits = np.asarray(upper_limits, dtype=np.float64)

        T = joints_all.shape[0]
        indices = list(range(0, T, frame_stride))
        if max_frames is not None:
            indices = indices[:max_frames]

        work_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = work_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        w2c = torch.tensor(extrinsics.astype(np.float32), device="cuda")
        device = torch.device("cuda")
        background_color = background_color.to(device)

        video_path = work_dir / "drrobot_render.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        for fi, t in enumerate(indices):
            q_norm = normalize_joints(joints_all[t], lower_limits, upper_limits)
            joint_t = torch.tensor(q_norm, dtype=torch.float32, device=device)

            cam = Camera_Pose(
                w2c,
                fov_x,
                fov_y,
                image_width,
                image_height,
                joint_pose=joint_t,
                zero_init=True,
            ).to(device)

            with torch.no_grad():
                rgb = render(cam, gaussians, background_color)["render"]
            rgb = torch.clamp(rgb, 0.0, 1.0)
            hwc = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
                np.uint8
            )
            bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)

            if writer is None:
                h, w = bgr.shape[:2]
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            writer.write(bgr)
            cv2.imwrite(str(frames_dir / f"frame_{fi:06d}.jpg"), bgr)

        if writer is not None:
            writer.release()

    finally:
        sys.argv = saved_argv
        os.chdir(prev_cwd)
        dr_str = str(drrobot_root)
        if dr_str in sys.path:
            sys.path.remove(dr_str)

    return video_path, indices


# ---------------------------------------------------------------------------
# Step 2: mask extraction (via Any4D helper)
# ---------------------------------------------------------------------------


def run_extract_mask(
    video: Path, frames_dir: Path, ref_idx: int, extract_fps: float | None
) -> Path:
    """Run Any4D's extract_frames_and_mask.py to generate a foreground mask."""
    script = ANY4D_ROOT / "scripts" / "extract_frames_and_mask.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    cmd = [
        sys.executable,
        str(script),
        "--video",
        str(video),
        "--output_dir",
        str(frames_dir),
        "--ref_idx",
        str(ref_idx),
    ]
    if extract_fps is not None:
        cmd += ["--fps", str(extract_fps)]
    subprocess.run(cmd, check=True, cwd=str(ANY4D_ROOT))
    return frames_dir / "ref_binary_mask.png"


# ---------------------------------------------------------------------------
# Step 3: flow extraction
# ---------------------------------------------------------------------------


def compute_flow_any4d(
    frames_dir: Path,
    mask_path: Path,
    checkpoint: Path,
    n_frames: int,
    output_flow: Path,
    ref_idx: int,
    target_h: int,
    target_w: int,
) -> Path:
    """Invoke extract_any4d_flow.py (subprocess) to get dense 2D flow from Any4D."""
    if not EXTRACT_FLOW_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Missing {EXTRACT_FLOW_SCRIPT}. Place extract_any4d_flow.py next to this script."
        )
    cmd = [
        sys.executable,
        str(EXTRACT_FLOW_SCRIPT),
        "--frames_dir",
        str(frames_dir),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_flow),
        "--start_idx",
        "0",
        "--end_idx",
        str(n_frames),
        "--ref_idx",
        str(ref_idx),
        "--target_height",
        str(target_h),
        "--target_width",
        str(target_w),
    ]
    if mask_path and mask_path.is_file():
        cmd += ["--mask_path", str(mask_path)]
    subprocess.run(cmd, check=True, cwd=str(ANY4D_ROOT))
    return output_flow


def compute_flow_opencv(
    frames_dir: Path, n_frames: int, output_flow: Path
) -> Path:
    """Reference-anchored Farneback optical flow on DrRobot rendered frames.

    flow[0] = zeros (reference), flow[t] = pixel displacement from frame 0 to t.
    Fast fallback when Any4D is unavailable.
    """
    import glob as globmod

    paths = sorted(globmod.glob(str(frames_dir / "frame_*.jpg")))[:n_frames]
    if len(paths) < 2:
        raise RuntimeError(f"Need >=2 frames in {frames_dir}, got {len(paths)}")

    ref_bgr = cv2.imread(paths[0])
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    h, w = ref_gray.shape[:2]
    T = len(paths)
    flow_all = np.zeros((T, h, w, 2), dtype=np.float32)

    for t in range(1, T):
        cur_gray = cv2.cvtColor(cv2.imread(paths[t]), cv2.COLOR_BGR2GRAY)
        flow_2d = cv2.calcOpticalFlowFarneback(
            ref_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_all[t] = flow_2d

    output_flow.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_flow), flow_all)
    print(f"  OpenCV flow {flow_all.shape} -> {output_flow}")
    return output_flow


# ---------------------------------------------------------------------------
# Step 4: assemble DROID real-world video
# ---------------------------------------------------------------------------


def assemble_droid_video(
    scene_dir: Path,
    indices: list[int],
    output_mp4: Path,
    fps: float,
) -> Path:
    """Assemble selected DROID images0/ frames into an mp4.

    ``indices`` are the joint/image timestep indices used during DrRobot
    rendering, ensuring temporal alignment between the DROID video and flow.
    """
    img_dir = scene_dir / "images0"
    all_imgs = {int(p.stem): p for p in img_dir.glob("*.jpg")}
    if not all_imgs:
        raise FileNotFoundError(f"No images in {img_dir}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    for t in indices:
        if t not in all_imgs:
            continue
        bgr = cv2.imread(str(all_imgs[t]))
        if writer is None:
            h, w = bgr.shape[:2]
            writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (w, h))
        writer.write(bgr)

    if writer is not None:
        writer.release()
    return output_mp4


# ---------------------------------------------------------------------------
# Step 5: metadata CSV for world-model training
# ---------------------------------------------------------------------------


def build_wm_metadata(
    output_root: Path,
    scenes_info: list[dict],
    csv_path: Path | None = None,
) -> Path:
    """Write metadata.csv with columns: video, prompt, flow.

    Paths are relative to ``output_root`` so the CSV is relocatable.
    """
    if csv_path is None:
        csv_path = output_root / "metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt", "flow"])
        writer.writeheader()
        for info in scenes_info:
            writer.writerow(
                {
                    "video": info["video_rel"],
                    "prompt": info["prompt"],
                    "flow": info["flow_rel"],
                }
            )
    print(f"\nWrote WM metadata CSV ({len(scenes_info)} rows) -> {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Per-scene orchestration
# ---------------------------------------------------------------------------


def process_scene(
    scene_dir: Path,
    output_dir: Path,
    drrobot_model_path: str,
    any4d_checkpoint: Path,
    fps: float,
    max_frames: int | None,
    frame_stride: int,
    ref_idx: int,
    skip_drrobot: bool,
    flow_method: str,
    image_width: int | None,
    image_height: int | None,
) -> dict | None:
    """Full pipeline for one scene.  Returns a dict for metadata CSV, or None on skip."""
    scene_name = scene_dir.name
    work_dir = output_dir / scene_name
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path = work_dir / "drrobot_render.mp4"
    frames_dir = work_dir / "frames"

    # -- Determine image dimensions for flow resizing ---
    if image_width and image_height:
        vid_w, vid_h = image_width, image_height
    else:
        vid_w, vid_h = get_scene_image_size(scene_dir)

    # ---- 1) DrRobot render ----
    if not skip_drrobot:
        print(f"  [1/5] DrRobot rendering ...")
        _, indices = render_scene_drrobot(
            scene_dir,
            work_dir,
            drrobot_model_path,
            fps=fps,
            max_frames=max_frames,
            frame_stride=frame_stride,
            image_width=vid_w,
            image_height=vid_h,
        )
    else:
        if not video_path.is_file():
            raise FileNotFoundError(
                f"No rendered video at {video_path}. Run without --skip_drrobot."
            )
        T = np.load(scene_dir / "joints.npy").shape[0]
        indices = list(range(0, T, frame_stride))
        if max_frames:
            indices = indices[:max_frames]

    # ---- 2) Extract mask ----
    print(f"  [2/5] Extracting foreground mask ...")
    mask_path = run_extract_mask(video_path, frames_dir, ref_idx, extract_fps=fps)

    # ---- 3) Compute flow ----
    import glob as globmod

    n_rendered = len(globmod.glob(str(frames_dir / "frame_*.jpg")))
    if n_rendered < 2:
        print(f"  WARNING: Only {n_rendered} frame(s), skipping scene.")
        return None

    flow_path = work_dir / "flow.npy"
    n_flow = min(n_rendered, max_frames) if max_frames else n_rendered

    if flow_method == "any4d":
        print(f"  [3/5] Any4D flow extraction ({n_flow} frames) ...")
        compute_flow_any4d(
            frames_dir,
            mask_path,
            any4d_checkpoint,
            n_flow,
            flow_path,
            ref_idx=ref_idx,
            target_h=vid_h,
            target_w=vid_w,
        )
    else:
        print(f"  [3/5] OpenCV Farneback flow ({n_flow} frames) ...")
        compute_flow_opencv(frames_dir, n_flow, flow_path)

    # ---- 4) Assemble DROID video ----
    droid_mp4 = work_dir / "droid_video.mp4"
    droid_img_dir = scene_dir / "images0"
    n_droid_imgs = len(list(droid_img_dir.glob("*.jpg")))
    n_joints = np.load(scene_dir / "joints.npy").shape[0]

    if n_droid_imgs == n_joints:
        print(f"  [4/5] Assembling DROID video ({len(indices)} frames) ...")
        assemble_droid_video(scene_dir, indices, droid_mp4, fps)
    elif n_droid_imgs > 0:
        print(
            f"  [4/5] Image/joint count mismatch ({n_droid_imgs} vs {n_joints}), "
            f"using all available images ..."
        )
        available = sorted(int(p.stem) for p in droid_img_dir.glob("*.jpg"))
        use_indices = available[:: frame_stride]
        if max_frames:
            use_indices = use_indices[:max_frames]
        assemble_droid_video(scene_dir, use_indices, droid_mp4, fps)
    else:
        print(f"  [4/5] No DROID images, using DrRobot render as video.")
        import shutil

        shutil.copy2(video_path, droid_mp4)

    # ---- 5) Prompt ----
    prompt = get_droid_prompt(scene_dir)
    prompt_file = work_dir / "prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    print(f"  [5/5] Prompt: {prompt!r}")

    video_rel = str(Path(scene_name) / "droid_video.mp4")
    flow_rel = str(Path(scene_name) / "flow.npy")

    return {"video_rel": video_rel, "prompt": prompt, "flow_rel": flow_rel}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DATASET_ROOT,
        help="Root of droid_extract_whole with scene_* subdirectories.",
    )
    parser.add_argument(
        "--drrobot_model_path",
        type=str,
        default=str(DRROBOT_ROOT / "output" / "panda"),
        help="Trained DrRobot model directory (must contain cfg_args, point_cloud/).",
    )
    parser.add_argument(
        "--any4d_checkpoint",
        type=Path,
        default=ANY4D_CHECKPOINT,
        help="Path to any4d_4v_combined.pth.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=SCRIPT_DIR / "droid_scene_outputs",
        help="Root output directory. Per-scene results go in <output_root>/scene_*/.",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Video FPS.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Use every k-th joint timestep for rendering.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=40,
        help="Max frames to render per scene.",
    )
    parser.add_argument(
        "--ref_idx", type=int, default=0, help="Reference frame index for masks."
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=None,
        help="Override render width (default: infer from scene images or 640).",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=None,
        help="Override render height (default: infer from scene images or 360).",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="*",
        default=None,
        help="Process only these scene names (e.g. scene_0 scene_1). Default: all.",
    )
    parser.add_argument(
        "--skip_drrobot",
        action="store_true",
        help="Skip DrRobot rendering (expects existing renders in output_root).",
    )
    parser.add_argument(
        "--flow_method",
        type=str,
        choices=["any4d", "opencv"],
        default="any4d",
        help="How to compute optical flow. 'any4d' projects 3D scene flow to 2D "
        "(accurate, needs GPU + Any4D). 'opencv' uses Farneback on the DrRobot "
        "renders (fast, no extra deps).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenes whose output directory already has flow.npy + droid_video.mp4.",
    )

    args = parser.parse_args()
    scenes = discover_scenes(args.dataset_root)
    if not scenes:
        print(f"No valid scenes found under {args.dataset_root}")
        sys.exit(1)

    if args.scenes:
        allowed = set(args.scenes)
        scenes = [s for s in scenes if s.name in allowed]

    print(f"Found {len(scenes)} scene(s) under {args.dataset_root}")
    print(f"Flow method: {args.flow_method}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    succeeded, failed = [], []
    scenes_info: list[dict] = []

    for i, scene_dir in enumerate(scenes):
        name = scene_dir.name
        print(f"\n[{i + 1}/{len(scenes)}] {name}")

        if args.resume:
            done = (
                (args.output_root / name / "flow.npy").is_file()
                and (args.output_root / name / "droid_video.mp4").is_file()
            )
            if done:
                print(f"  Skipping (already complete)")
                prompt = get_droid_prompt(scene_dir)
                scenes_info.append(
                    {
                        "video_rel": str(Path(name) / "droid_video.mp4"),
                        "prompt": prompt,
                        "flow_rel": str(Path(name) / "flow.npy"),
                    }
                )
                succeeded.append(name)
                continue

        try:
            info = process_scene(
                scene_dir=scene_dir,
                output_dir=args.output_root,
                drrobot_model_path=args.drrobot_model_path,
                any4d_checkpoint=args.any4d_checkpoint.resolve(),
                fps=args.fps,
                max_frames=args.max_frames,
                frame_stride=args.frame_stride,
                ref_idx=args.ref_idx,
                skip_drrobot=args.skip_drrobot,
                flow_method=args.flow_method,
                image_width=args.image_width,
                image_height=args.image_height,
            )
            if info is not None:
                scenes_info.append(info)
            succeeded.append(name)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((name, str(e)))

    # ---- Write WM metadata CSV ----
    if scenes_info:
        build_wm_metadata(args.output_root, scenes_info)

    print("\n" + "=" * 60)
    print(
        f"Summary: {len(succeeded)} succeeded, {len(failed)} failed "
        f"out of {len(scenes)} scenes"
    )
    if scenes_info:
        print(f"WM training CSV: {args.output_root / 'metadata.csv'}")
        print(
            f"  Use with:  --dataset_base_path {args.output_root} "
            f"--metadata_csv {args.output_root / 'metadata.csv'}"
        )
    if failed:
        print("Failed scenes:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
