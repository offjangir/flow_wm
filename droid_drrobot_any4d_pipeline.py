#!/usr/bin/env python3
"""
End-to-end helper for DROID raw trajectories → DrRobot render → Any4D 3D point tracks.

DROID layout (per episode directory):
  - trajectory.h5
  - metadata*.json
  - recordings/MP4/*.mp4

Camera extrinsics in HDF5 follow the Rerun DROID example: translation (m) + Euler xyz (rad):
  https://github.com/rerun-io/python-example-droid-dataset/blob/master/src/raw.py

Prerequisites:
  - GPU, CUDA PyTorch, DrRobot deps (gsplat, mujoco with EGL typical on headless servers)
  - Any4D installed per Any4D/README.md and checkpoint at --any4d_checkpoint
  - Set MUJOCO_GL=egl (or osmesa) if you have no display

Example:
  export MUJOCO_GL=egl
  python droid_drrobot_any4d_pipeline.py \\
    --trajectory_dir /data/group_data/katefgroup/datasets/droid_chenyu/data/droid_raw/1.0.1/CLVR/success/2023-06-18/Mon_Jun_19_00:02:50_2023 \\
    --drrobot_model_path /path/to/trained/drrobot/output \\
    --any4d_checkpoint /path/to/any4d_4v_combined.pth \\
    --camera_role ext1_left \\
    --max_frames 40

This writes:
  <work_dir>/drrobot_render.mp4
  <work_dir>/frames/frame_*.jpg
  <work_dir>/ref_binary_mask.png
  <work_dir>/point_tracks.npz   (Any4D subsampled 3D points per time step; keys t0000, t0001, ...)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_droid_metadata(trajectory_dir: Path) -> dict:
    jsons = list(trajectory_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No metadata JSON in {trajectory_dir}")
    with open(jsons[0], "r", encoding="utf-8") as f:
        return json.load(f)


def _camera_h5_key(metadata: dict, role: str) -> str:
    role = role.lower().strip()
    mapping = {
        "wrist_left": ("wrist_cam_serial", "_left"),
        "ext1_left": ("ext1_cam_serial", "_left"),
        "ext1_right": ("ext1_cam_serial", "_right"),
        "ext2_left": ("ext2_cam_serial", "_left"),
        "ext2_right": ("ext2_cam_serial", "_right"),
    }
    if role not in mapping:
        raise ValueError(f"camera_role must be one of {list(mapping)}")
    field, suffix = mapping[role]
    serial = metadata[field]
    return f"{serial}{suffix}"


def droid_extrinsic6_to_w2c(ext6) -> "object":
    """DROID 6-vector → 4x4 world-to-camera (numpy), consistent with inv(c2w)."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    t = np.asarray(ext6[:3], dtype=np.float64)
    R_wc = Rotation.from_euler("xyz", np.asarray(ext6[3:], dtype=np.float64)).as_matrix()
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_wc
    c2w[:3, 3] = t
    return np.linalg.inv(c2w).astype(np.float32)


def map_droid_joints_to_drrobot_normalized(
    joint_rad: "object", lower_limits: "object", upper_limits: "object"
) -> "object":
    """Map real joint angles to [-1, 1] (robot_dataset.py); pad/truncate to kinematic chain length."""
    import numpy as np

    q = np.asarray(joint_rad, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower_limits, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper_limits, dtype=np.float64).reshape(-1)
    n_chain = len(lo)
    out = np.zeros(n_chain, dtype=np.float32)
    use = min(len(q), n_chain)
    if use == 0:
        return out
    lo_s, hi_s, q_s = lo[:use], hi[:use], q[:use]
    scale = 2.0 / (hi_s - lo_s + 1e-8)
    out[:use] = np.clip((q_s - lo_s) * scale - 1.0, -1.0, 1.0).astype(np.float32)
    return out


def map_droid_state_to_drrobot_normalized(
    joint_rad: "object",
    gripper_pos: float | None,
    lower_limits: "object",
    upper_limits: "object",
) -> "object":
    """
    Map DROID robot state to DrRobot normalized joint vector.

    For Panda-like DROID data (7 arm joints + scalar gripper_position) rendered with
    DrRobot panda_2f85 (13 DoFs), map:
      - arm joints -> indices [0..6]
      - gripper scalar -> driver joints [7, 10]
    Remaining gripper mechanism joints are kept at 0.
    """
    import numpy as np

    out = map_droid_joints_to_drrobot_normalized(joint_rad, lower_limits, upper_limits)
    lo = np.asarray(lower_limits, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper_limits, dtype=np.float64).reshape(-1)
    q = np.asarray(joint_rad, dtype=np.float64).reshape(-1)

    # Panda + Robotiq style chain in this setup has 13 DoFs:
    # 7 arm joints + 6 gripper mechanism joints.
    if gripper_pos is not None and len(lo) >= 13 and len(q) == 7:
        g = float(gripper_pos)
        driver_indices = [7, 10]
        for idx in driver_indices:
            lo_i = lo[idx]
            hi_i = hi[idx]
            if 0.0 <= g <= 1.0:
                # Common DROID convention: scalar open fraction in [0, 1].
                q_i = lo_i + g * (hi_i - lo_i)
            else:
                # Fallback if the input is already in joint units.
                q_i = min(max(g, lo_i), hi_i)
            out[idx] = np.clip((q_i - lo_i) * (2.0 / (hi_i - lo_i + 1e-8)) - 1.0, -1.0, 1.0)

    return out.astype(np.float32)


def render_drrobot_trajectory(
    trajectory_dir: Path,
    drrobot_root: Path,
    drrobot_model_path: str,
    work_dir: Path,
    camera_h5_key: str,
    fps: float,
    max_frames: int | None,
    frame_stride: int,
    camera_fallback_if_white: bool = True,
) -> Path:
    """Render episode with DrRobot; save MP4 + JPG frames. Returns path to MP4."""
    import h5py
    import numpy as np
    import torch
    import cv2

    traj_path = trajectory_dir / "trajectory.h5"
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)

    h5_cam = f"observation/camera_extrinsics/{camera_h5_key}"
    h5_joints = "observation/robot_state/joint_positions"
    h5_gripper = "observation/robot_state/gripper_position"

    saved_argv = sys.argv[:]
    prev_cwd = os.getcwd()
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
        ref_cam = sample_cameras[0]
        sample_w2c = (
            ref_cam.world_view_transform.detach().cpu().transpose(0, 1).numpy().astype(np.float32)
        )
        lower_limits, upper_limits = gaussians.chain.get_joint_limits()
        lower_limits = np.asarray(lower_limits, dtype=np.float64)
        upper_limits = np.asarray(upper_limits, dtype=np.float64)

        with h5py.File(traj_path, "r") as f:
            if h5_cam not in f:
                keys = [k for k in f["observation/camera_extrinsics"].keys()]
                raise KeyError(f"Missing {h5_cam}. Available camera_extrinsics keys (sample): {keys[:12]}...")
            extr = f[h5_cam][:]
            joints = f[h5_joints][:]
            gripper = f[h5_gripper][:] if h5_gripper in f else None

        if extr.ndim != 2 or extr.shape[1] != 6:
            raise ValueError(f"Expected camera extrinsics shape [T,6], got {extr.shape} from {h5_cam}")
        if joints.ndim != 2:
            raise ValueError(f"Expected joint positions shape [T,J], got {joints.shape} from {h5_joints}")
        if gripper is not None and gripper.ndim != 1:
            raise ValueError(f"Expected gripper_position shape [T], got {gripper.shape} from {h5_gripper}")

        chain_joints = int(lower_limits.shape[0])
        data_joints = int(joints.shape[1])
        print(
            f"[Shapes] extrinsics={extr.shape}, joints={joints.shape}, "
            f"drrobot_chain_joints={chain_joints}"
        )
        if gripper is not None:
            print(f"[Shapes] gripper_position={gripper.shape}")
        if data_joints != chain_joints:
            print(
                f"[Warn] Joint dim mismatch: DROID has {data_joints}, DrRobot chain expects {chain_joints}. "
                "Pipeline will truncate/pad via map_droid_joints_to_drrobot_normalized."
            )
        if gripper is not None and chain_joints >= 13 and data_joints == 7:
            print(
                "[Map] Using DROID gripper_position to drive DrRobot Panda gripper driver joints "
                "(indices 7 and 10)."
            )

        n = min(extr.shape[0], joints.shape[0])
        if gripper is not None:
            n = min(n, int(gripper.shape[0]))
        indices = list(range(0, n, frame_stride))
        if max_frames is not None:
            indices = indices[:max_frames]

        work_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = work_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        video_path = work_dir / "drrobot_render.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        device = torch.device("cuda")
        background_color = background_color.to(device)
        use_sample_camera_pose = False

        def render_bgr(w2c_mat: "object", q_norm: "object") -> "object":
            w2c_t = torch.tensor(w2c_mat, dtype=torch.float32, device=device)
            joint_t = torch.tensor(q_norm, dtype=torch.float32, device=device)
            cam = Camera_Pose(
                w2c_t,
                ref_cam.FoVx,
                ref_cam.FoVy,
                int(ref_cam.image_width),
                int(ref_cam.image_height),
                joint_pose=joint_t,
                zero_init=True,
            ).to(device)
            with torch.no_grad():
                rgb = render(cam, gaussians, background_color)["render"]
            rgb = torch.clamp(rgb, 0.0, 1.0)
            hwc = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            return cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)

        for fi, t in enumerate(indices):
            g_t = float(gripper[t]) if gripper is not None else None
            q_norm = map_droid_state_to_drrobot_normalized(joints[t], g_t, lower_limits, upper_limits)
            if use_sample_camera_pose:
                bgr = render_bgr(sample_w2c, q_norm)
            else:
                w2c = droid_extrinsic6_to_w2c(extr[t])
                bgr = render_bgr(w2c, q_norm)
                # Guardrail: some episodes/camera keys can produce near-blank views.
                if fi == 0 and camera_fallback_if_white:
                    mean_v = float(bgr.mean())
                    std_v = float(bgr.std())
                    if mean_v > 245.0 and std_v < 5.0:
                        print(
                            "[Warn] First rendered frame is near-white "
                            f"(mean={mean_v:.2f}, std={std_v:.2f}). "
                            "Falling back to DrRobot sample camera pose."
                        )
                        use_sample_camera_pose = True
                        bgr = render_bgr(sample_w2c, q_norm)
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
        dr_path = str(drrobot_root.resolve())
        if dr_path in sys.path:
            sys.path.remove(dr_path)

    return video_path


def run_extract_mask(any4d_root: Path, video: Path, frames_dir: Path, ref_idx: int, extract_fps: float) -> Path:
    script = any4d_root / "scripts" / "extract_frames_and_mask.py"
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
        "--fps",
        str(extract_fps),
    ]
    subprocess.run(cmd, check=True, cwd=str(any4d_root))
    return frames_dir / "ref_binary_mask.png"


def run_any4d(
    any4d_root: Path,
    frames_dir: Path,
    mask_path: Path,
    checkpoint: Path,
    start_idx: int,
    end_idx: int,
    ref_idx: int,
    tracks_out: Path,
    extra_demo_args: list[str],
) -> None:
    script = any4d_root / "scripts" / "demo_inference.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    cmd = [
        sys.executable,
        str(script),
        "--video_images_folder_path",
        str(frames_dir),
        "--checkpoint_path",
        str(checkpoint),
        "--start_idx",
        str(start_idx),
        "--end_idx",
        str(end_idx),
        "--ref_img_idx",
        str(ref_idx),
        "--ref_img_binary_mask_path",
        str(mask_path),
        "--viz",
        "--connect",
        "false",
    ]
    # Some Any4D versions expose a custom save arg, others do not.
    help_cmd = [sys.executable, str(script), "-h"]
    help_out = subprocess.run(help_cmd, check=False, capture_output=True, text=True, cwd=str(any4d_root))
    if "--save_point_tracks_npz" in (help_out.stdout or ""):
        cmd.extend(["--save_point_tracks_npz", str(tracks_out)])
    else:
        print(
            "[Info] Any4D demo_inference.py does not support --save_point_tracks_npz; "
            "running without explicit NPZ export flag."
        )
    cmd.extend(extra_demo_args)
    subprocess.run(cmd, check=True, cwd=str(any4d_root))


def main() -> None:
    sidegig = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trajectory_dir",
        type=Path,
        required=True,
        help="DROID episode folder (contains trajectory.h5, metadata JSON, recordings/).",
    )
    parser.add_argument(
        "--drrobot_root",
        type=Path,
        default=sidegig / "drrobot",
        help="DrRobot repository root (contains video_api.py, scene/, …).",
    )
    parser.add_argument(
        "--drrobot_model_path",
        type=str,
        required=True,
        help="Trained DrRobot model directory (must contain cfg_args and point_cloud/).",
    )
    parser.add_argument(
        "--any4d_root",
        type=Path,
        default=sidegig / "Any4D",
        help="Any4D repository root.",
    )
    parser.add_argument(
        "--any4d_checkpoint",
        type=Path,
        required=True,
        help="Path to any4d_4v_combined.pth (or compatible).",
    )
    parser.add_argument(
        "--camera_role",
        type=str,
        default="ext1_left",
        choices=["wrist_left", "ext1_left", "ext1_right", "ext2_left", "ext2_right"],
        help="Which DROID camera to render (resolved via episode metadata serials).",
    )
    parser.add_argument("--work_dir", type=Path, default=None, help="Output directory (default: trajectory_dir/any4d_drrobot_work).")
    parser.add_argument("--fps", type=float, default=15.0, help="FPS for rendered MP4 and frame extraction.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Use every k-th trajectory timestep.")
    parser.add_argument(
        "--disable_white_fallback",
        action="store_true",
        help="Disable auto fallback to DrRobot sample camera when first render is near-white.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Cap number of rendered frames. Use -1 for full trajectory.",
    )
    parser.add_argument("--ref_idx", type=int, default=0, help="Reference frame index for MoGe / scene-flow mask.")
    parser.add_argument(
        "--skip_drrobot",
        action="store_true",
        help="Skip rendering; expect frames + drrobot_render.mp4 already in work_dir.",
    )
    parser.add_argument(
        "--skip_any4d",
        action="store_true",
        help="Only run DrRobot rendering and frame/mask extraction.",
    )
    parser.add_argument("demo_extra", nargs="*", help="Extra args passed to demo_inference.py (quote as needed).")

    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        args.max_frames = None

    if args.work_dir is None:
        args.work_dir = args.trajectory_dir / "any4d_drrobot_work"
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_droid_metadata(args.trajectory_dir)
    cam_key = _camera_h5_key(meta, args.camera_role)

    if not args.skip_drrobot:
        print(f"=== DrRobot render: camera_extrinsics/{cam_key} ===")
        render_drrobot_trajectory(
            args.trajectory_dir.resolve(),
            args.drrobot_root.resolve(),
            args.drrobot_model_path,
            args.work_dir,
            cam_key,
            fps=args.fps,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            camera_fallback_if_white=not args.disable_white_fallback,
        )
    video = args.work_dir / "drrobot_render.mp4"
    frames_dir = args.work_dir / "frames"
    if not video.is_file():
        raise FileNotFoundError(f"Missing rendered video {video} (remove --skip_drrobot or render first).")

    print("=== extract_frames_and_mask ===")
    mask_path = run_extract_mask(
        args.any4d_root.resolve(),
        video,
        frames_dir,
        ref_idx=args.ref_idx,
        extract_fps=args.fps,
    )

    if args.skip_any4d:
        print(f"Done (Any4D skipped). Frames: {frames_dir}, mask: {mask_path}")
        return

    n_jpg = len(glob.glob(str(frames_dir / "frame_*.jpg")))
    if n_jpg < 2:
        raise RuntimeError(f"Need at least 2 frames in {frames_dir}")

    start_idx = 0
    end_idx = n_jpg if args.max_frames is None else min(n_jpg, args.max_frames)
    ref_idx = max(0, min(args.ref_idx, end_idx - 1))

    tracks_npz = args.work_dir / "point_tracks.npz"
    print("=== Any4D demo_inference (3D tracks) ===")
    run_any4d(
        args.any4d_root.resolve(),
        frames_dir,
        mask_path,
        args.any4d_checkpoint.resolve(),
        start_idx=start_idx,
        end_idx=end_idx,
        ref_idx=ref_idx,
        tracks_out=tracks_npz,
        extra_demo_args=list(args.demo_extra),
    )
    print(f"Outputs under {args.work_dir}")


if __name__ == "__main__":
    main()
