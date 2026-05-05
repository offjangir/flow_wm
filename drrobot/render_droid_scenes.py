"""Render DROID ``scene_*`` folders through a trained DrRobot checkpoint (CLI: ``main()``).

For DROID data in this repo (``trajectory.h5`` extrinsics + ``metadata_*.json``), the intended
setup is the default: **do not** pass ``--gl_flip``. That matches the real camera view. Only use
``--gl_flip`` if the image is clearly upside-down or backwards after confirming ``--camera_key``
and ``--trajectory_camera`` are correct.

Output **``--fps``** defaults to ``match``: frame rate is read from ``recordings/MP4/<serial>.mp4``
for ``--camera_key`` (often 60), so rendered clip duration matches the real recording. Passing
``--fps 15`` still works for fixed-rate exports.
"""

import argparse
import configparser
import copy
import glob
import json
import os
import subprocess
import urllib.request
from argparse import Namespace
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation
try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    ImageSequenceClip = None

from arguments import OptimizationParams
from gaussian_renderer import GaussianModel, render
from scene import RobotScene
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from utils.mujoco_utils import compute_camera_extrinsic_matrix

# ---------------------------------------------------------------------------
# ZED factory intrinsics (from Stereolabs calibration server, 720p left eye)
# Keyed by camera serial number.  To add a new camera, run:
#   curl -L "https://www.stereolabs.com/developers/calib/?SN=<SERIAL>"
# and copy the [LEFT_CAM_HD] section values here.
# ---------------------------------------------------------------------------

ZED_INTRINSICS_720P: Dict[str, Dict[str, float]] = {
    # wrist  -- ZED 2i
    "16787047": {"fx": 771.21, "fy": 771.04, "cx": 646.12, "cy": 365.87},
    # ext1   -- ZED 2i
    "20103212": {"fx": 533.41, "fy": 533.46, "cx": 646.49, "cy": 360.19},
    # ext2   -- ZED 2i
    "20655732": {"fx": 533.73, "fy": 533.62, "cx": 638.15, "cy": 355.50},
}

ZED_DEFAULT_INTRINSICS_720P: Dict[str, float] = {
    "fx": 533.0, "fy": 533.0, "cx": 640.0, "cy": 360.0,
}

_STEREOLABS_URL = "https://www.stereolabs.com/developers/calib/?SN={serial}"


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if code == 308:
            code = 301
        return super().redirect_request(req, fp, code, msg, headers, newurl)

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, code, msg, headers)


_url_opener = urllib.request.build_opener(_RedirectHandler)


def _fetch_zed_intrinsics_online(serial: str) -> Optional[Dict[str, float]]:
    """Try to fetch 720p left-eye intrinsics from Stereolabs for an unknown serial."""
    url = _STEREOLABS_URL.format(serial=serial)
    try:
        with _url_opener.open(url, timeout=10) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        print(f"[intrinsics] online fetch failed for serial {serial}: {e}")
        return None

    cp = configparser.ConfigParser()
    cp.read_string(text)
    section = "LEFT_CAM_HD"
    if not cp.has_section(section):
        return None
    d = {}
    for key in ("fx", "fy", "cx", "cy"):
        if cp.has_option(section, key):
            d[key] = float(cp.get(section, key))
    return d if d else None


def get_scene_camera_intrinsics(
    scene_dir: str, camera_key: str,
) -> Dict[str, float]:
    """Return ZED intrinsics ``{fx, fy, cx, cy}`` for a scene's camera at 720p.

    Lookup order:
      1. Hardcoded ``ZED_INTRINSICS_720P`` (known serials)
      2. Online fetch from Stereolabs (new serial -- result is cached in the dict for the run)
      3. ``ZED_DEFAULT_INTRINSICS_720P`` (sensible ZED 2i defaults)
    """
    if camera_key == "default":
        return ZED_DEFAULT_INTRINSICS_720P

    md = load_scene_metadata(scene_dir)
    if md is None:
        print(f"[intrinsics] no metadata for {os.path.basename(scene_dir)}, using defaults")
        return ZED_DEFAULT_INTRINSICS_720P

    try:
        serial = camera_serial_for_key(md, camera_key)
    except KeyError:
        print(f"[intrinsics] no serial for {camera_key} in metadata, using defaults")
        return ZED_DEFAULT_INTRINSICS_720P

    if serial in ZED_INTRINSICS_720P:
        return ZED_INTRINSICS_720P[serial]

    print(f"[intrinsics] serial {serial} not in hardcoded table, trying online...")
    fetched = _fetch_zed_intrinsics_online(serial)
    if fetched is not None:
        ZED_INTRINSICS_720P[serial] = fetched
        print(f"[intrinsics] fetched & cached: serial={serial} fx={fetched['fx']:.2f} fy={fetched['fy']:.2f}")
        return fetched

    print(f"[intrinsics] using defaults for unknown serial {serial}")
    return ZED_DEFAULT_INTRINSICS_720P


def _local_trained_run_hints() -> str:
    """If this script lives inside the drrobot repo, suggest output/* runs that have cfg_args."""
    repo_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    if not os.path.isdir(repo_output):
        return ""
    runs = sorted(glob.glob(os.path.join(repo_output, "*", "cfg_args")))
    if not runs:
        return ""
    dirs = [os.path.dirname(p) for p in runs]
    return "\n\nRuns next to this repo with cfg_args:\n  " + "\n  ".join(dirs)


def _parse_r_frame_rate(rate: str) -> Optional[float]:
    """Parse ffprobe ``r_frame_rate`` like ``60/1`` or ``30000/1001``."""
    s = (rate or "").strip()
    if not s or s == "N/A":
        return None
    if "/" in s:
        num, den = s.split("/", 1)
        try:
            d = float(den)
            return float(num) / d if d else None
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def recording_fps_for_camera(scene_dir: str, camera_key: Optional[str]) -> Optional[float]:
    """Read FPS from ``recordings/MP4/<serial>.mp4`` for this camera (matches DROID playback speed)."""
    if not camera_key or camera_key == "default":
        return None
    try:
        md = load_scene_metadata(scene_dir)
        if md is None:
            return None
        serial = camera_serial_for_key(md, camera_key)
    except Exception:
        return None
    mp4_dir = os.path.join(scene_dir, "recordings", "MP4")
    candidates = [
        os.path.join(mp4_dir, f"{serial}.mp4"),
        os.path.join(mp4_dir, f"{serial}-stereo.mp4"),
    ]
    mp4_path = next((p for p in candidates if os.path.isfile(p)), None)
    if mp4_path is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=nw=1:nk=1",
        mp4_path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return _parse_r_frame_rate(out)


def write_video(frames, output_path: str, fps: Union[int, float]):
    if ImageSequenceClip is not None:
        clip = ImageSequenceClip(frames, fps=float(fps))
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
        return

    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise ImportError(
            "Neither moviepy nor imageio is available. Install one with:\n"
            "pip install moviepy\n"
            "or\n"
            "pip install imageio imageio-ffmpeg"
        ) from e

    with imageio.get_writer(output_path, fps=float(fps), codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def load_cfg_args(model_path: str) -> Namespace:
    model_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"--model_path is not a directory (path missing or wrong): {model_path}\n"
            "Pass the directory train.py used as model_path (it must contain cfg_args). "
            "Do not use README example names unless that folder exists on disk."
            + _local_trained_run_hints()
        )
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"Missing cfg_args in model directory:\n  {cfg_path}\n\n"
            "DrRobot writes this file when you run train.py (see prepare_output_and_logger). "
            "Pass the full path to your trained run (directory with cfg_args, sample_cameras.pkl, "
            "point_cloud/, robot_xml/)."
            + _local_trained_run_hints()
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_string = f.read()
    return eval(cfg_string)


def load_model(model_path: str, background: str = "white"):
    model_path = os.path.abspath(os.path.expanduser(model_path))
    cfg_args = load_cfg_args(model_path)
    cfg_args.model_path = model_path

    parser = argparse.ArgumentParser()
    opt_params = OptimizationParams(parser)

    sh_degree = getattr(cfg_args, "sh_degree", 3)
    gaussians = GaussianModel(sh_degree, opt_params)
    scene = RobotScene(cfg_args, gaussians, opt_params=opt_params, from_ckpt=True, load_iteration=-1)
    gaussians.model_path = scene.model_path

    if background == "black":
        bg_rgb = (0.0, 0.0, 0.0)
    else:
        bg_rgb = (1.0, 1.0, 1.0)
    bg_color = torch.tensor(bg_rgb, dtype=torch.float32, device="cuda")
    sample_cameras = scene.getSampleCameras(stage="pose_conditioned")
    if len(sample_cameras) == 0:
        raise RuntimeError("No sample cameras found in loaded checkpoint.")
    return gaussians, bg_color, sample_cameras


_FLIP_X_180 = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float64)


def droid_6dof_to_w2c(values, gl_flip: bool = False) -> np.ndarray:
    """Convert DROID [x, y, z, roll, pitch, yaw] -> 4x4 W2C for the Gaussian renderer.

    Steps (following DROID/r2d2 convention):
      1. Build T_base_cam (C2W) from position + ``from_euler('xyz', euler)``
      2. Optionally apply 180-deg X-axis flip (OpenCV -> OpenGL camera frame)
      3. Invert to get W2C for ``apply_extrinsic_to_camera``
    """
    v = np.asarray(values, dtype=np.float64).reshape(6)
    pos = v[:3]
    euler = v[3:]

    rot = ScipyRotation.from_euler("xyz", euler).as_matrix()

    T_base_cam = np.eye(4, dtype=np.float64)
    T_base_cam[:3, :3] = rot
    T_base_cam[:3, 3] = pos

    if gl_flip:
        T_base_cam = T_base_cam @ _FLIP_X_180

    T_w2c = np.linalg.inv(T_base_cam).astype(np.float32)

    cam_center = pos
    fwd = rot[:, 2] if not gl_flip else (T_base_cam[:3, :3])[:, 2]
    print(f"[droid_6dof_to_w2c] pos={cam_center}  fwd={fwd}  gl_flip={gl_flip}")

    return T_w2c


def mujoco_orbit_extrinsic(
    azimuth: float = 0.0, elevation: float = -45.0, distance: float = 3.0,
    lookat: tuple = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build a W2C extrinsic from MuJoCo-style orbit parameters (same as gradio_app_realtime.py)."""
    class _Cam:
        pass
    cam = _Cam()
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    cam.lookat = np.array(lookat, dtype=np.float64)
    return compute_camera_extrinsic_matrix(cam).astype(np.float32)


def load_scene_metadata(scene_dir: str) -> Optional[dict]:
    # Match both DROID's `metadata_<uuid>.json` and the simpler `metadata.json`
    # written by scripts/download_droid_5k.py.
    paths = (
        glob.glob(os.path.join(scene_dir, "metadata_*.json"))
        + glob.glob(os.path.join(scene_dir, "metadata.json"))
    )
    if not paths:
        return None
    with open(paths[0], "r", encoding="utf-8") as f:
        return json.load(f)


def camera_serial_for_key(metadata: dict, camera_key: str) -> str:
    field = f"{camera_key}_cam_serial"
    if field not in metadata:
        raise KeyError(f"Metadata missing {field!r} (keys like wrist_cam_serial, ext1_cam_serial, ...).")
    return str(metadata[field])


def parse_camera_extrinsic(scene_dir: str, camera_key: Optional[str], gl_flip: bool = False) -> Optional[np.ndarray]:
    candidates = [
        os.path.join(scene_dir, "camera_extrinsics.npy"),
        os.path.join(scene_dir, "extrinsics.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            arr = np.load(path)
            if arr.shape == (4, 4):
                return arr.astype(np.float32)
            if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                return arr[0].astype(np.float32)
            raise ValueError(f"Unsupported extrinsics shape in {path}: {arr.shape}")

    # Training checkpoint sample camera only (no DROID metadata / trajectory override).
    if camera_key == "default":
        return None

    metadata_paths = (
        glob.glob(os.path.join(scene_dir, "metadata_*.json"))
        + glob.glob(os.path.join(scene_dir, "metadata.json"))
    )
    if len(metadata_paths) == 0 or not camera_key:
        return None

    with open(metadata_paths[0], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    field = f"{camera_key}_cam_extrinsics"
    if field not in metadata:
        return None
    values = metadata[field]
    if not isinstance(values, list) or len(values) != 6:
        return None

    return droid_6dof_to_w2c(values, gl_flip=gl_flip)


def override_camera_intrinsics(
    camera, width: int, height: int,
    fov_x_deg: Optional[float] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
):
    """Override resolution and optionally FOV.

    Priority: explicit ``fov_x_deg`` > ``fx``/``fy`` from ZED intrinsics > keep training FOV.
    """
    camera.image_width = width
    camera.image_height = height

    if fov_x_deg is not None:
        camera.FoVx = float(np.radians(fov_x_deg))
    elif fx is not None:
        camera.FoVx = float(2.0 * np.arctan(width / (2.0 * fx)))

    if fy is not None and fov_x_deg is None:
        camera.FoVy = float(2.0 * np.arctan(height / (2.0 * fy)))
    else:
        camera.FoVy = float(
            2.0 * np.arctan(np.tan(camera.FoVx / 2.0) * height / width)
        )


def apply_extrinsic_to_camera(camera, extrinsic_matrix: np.ndarray):
    rotation_world_to_camera = extrinsic_matrix[:3, :3].astype(np.float32)
    translation_world_to_camera = extrinsic_matrix[:3, 3].astype(np.float32)

    camera.R = rotation_world_to_camera.T
    camera.T = translation_world_to_camera

    w2v = getWorld2View2(camera.R, camera.T, camera.trans, camera.scale)
    camera.world_view_transform = torch.tensor(w2v, device=camera.data_device).transpose(0, 1)
    camera.projection_matrix = getProjectionMatrix(
        znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx, fovY=camera.FoVy
    ).transpose(0, 1).to(camera.data_device)
    camera.full_proj_transform = (
        camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera.camera_center = camera.world_view_transform.inverse()[3, :3]


def _read_float_dataset(h5: h5py.Group, path: str) -> Optional[np.ndarray]:
    """Load a 1D/2D numeric dataset as float32, or None if missing or not a Dataset.

    HDF5 root satisfies ``"/" in h5`` and ``h5["/"]`` is a Group; ``np.array`` on that Group
    fails with errors like "could not convert string to float: 'action'". Guard against
    empty or root-only paths and non-Dataset objects.
    """
    p = (path or "").strip()
    if not p or p == "/":
        return None
    p = p.lstrip("/")
    if not p or p not in h5:
        return None
    obj = h5[p]
    if not isinstance(obj, h5py.Dataset):
        return None
    return np.asarray(obj[()], dtype=np.float32)


def load_trajectory_camera_extrinsics(
    scene_dir: str, serial: str, stereo: str
) -> Tuple[np.ndarray, str]:
    """Load per-frame [tx, ty, tz, rx, ry, rz] for one ZED serial + stereo eye from trajectory.h5."""
    h5_path = os.path.join(scene_dir, "trajectory.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    ds_name = f"{serial}_{stereo}"
    ds_path = f"observation/camera_extrinsics/{ds_name}"
    with h5py.File(h5_path, "r") as f:
        arr = _read_float_dataset(f, ds_path)
        if arr is None and stereo != "left":
            ds_path_fallback = f"observation/camera_extrinsics/{serial}_left"
            arr = _read_float_dataset(f, ds_path_fallback)
            if arr is not None:
                ds_path = ds_path_fallback
        if arr is None:
            if "observation" in f and "camera_extrinsics" in f["observation"]:
                avail = sorted(f["observation"]["camera_extrinsics"].keys())
            else:
                avail = []
            raise KeyError(
                f"No dataset {ds_path!r} in {h5_path}. "
                f"Available observation/camera_extrinsics keys (sample): {avail[:20]}{'...' if len(avail) > 20 else ''}"
            )
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"Expected camera extrinsics [T, 6], got {arr.shape} at {ds_path}")
        return arr.astype(np.float32), ds_path


def normalize_joints_from_chain(joints: np.ndarray, gaussians) -> np.ndarray:
    """Map raw radians to [-1, 1] using limits from the loaded kinematic chain (gaussians.chain).

    Same formula as ``RobotDataset.normalize_joint_positions`` and ``lrs`` denormalize
    (``lbs.py`` line 314: ``pose = (pose + 1) * (upper - lower) / 2 + lower``), just inverted.
    No external pickle needed — the chain already has the limits from the robot URDF/XML.
    """
    if joints.ndim != 2:
        raise ValueError(f"Expected joints with shape [T, J], got {joints.shape}")

    if np.abs(joints).max() <= 1.0 + 1e-3:
        print("[normalize_joints] joints already in ~[-1,1], skipping normalization.")
        return joints.astype(np.float32)

    lower, upper = gaussians.chain.get_joint_limits()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    n_arm = joints.shape[1]
    lower = lower[:n_arm]
    upper = upper[:n_arm]

    scale = 2.0 / (upper - lower)
    out = ((joints - lower) * scale - 1.0).astype(np.float32)
    print(f"[normalize_joints] radians -> [-1,1] via chain.get_joint_limits() ({n_arm} DOF)")
    print(f"[normalize_joints]   lower={lower}")
    print(f"[normalize_joints]   upper={upper}")
    print(f"[normalize_joints]   sample t=0 raw={joints[0]} -> norm={out[0]}")
    return out


def load_joints_and_gripper(scene_dir: str):
    # joints_path = os.path.join(scene_dir, "joints.npy")
    # if os.path.exists(joints_path):
    #     return np.load(joints_path), None
 
    h5_path = os.path.join(scene_dir, "trajectory.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing {h5_path}")

    joint_key = "observation/robot_state/joint_positions"
    gripper_key = "observation/robot_state/gripper_position"

    with h5py.File(h5_path, "r") as f:
        joints = _read_float_dataset(f, joint_key)
        gripper = _read_float_dataset(f, gripper_key)

        print(f"[load_joints_and_gripper] file={h5_path}")
        print(f"[load_joints_and_gripper] {joint_key}: shape={None if joints is None else joints.shape} dtype={None if joints is None else joints.dtype}")
        if joints is not None:
            print(f"[load_joints_and_gripper] {joint_key} t=0: {joints[0]}")
        print(
            f"[load_joints_and_gripper] {gripper_key}: shape={None if gripper is None else gripper.shape} dtype={None if gripper is None else gripper.dtype}"
        )
        if gripper is not None:
            print(f"[load_joints_and_gripper] {gripper_key} t=0: {gripper[0]}")

        if joints is None:
            raise KeyError(f"Dataset missing or not numeric: {joint_key} in {h5_path}")
        return joints, gripper


def render_scene_video(
    scene_dir: str,
    output_path: str,
    gaussians,
    bg_color,
    base_camera,
    fps: Union[int, float],
    camera_key: Optional[str],
    use_trajectory_camera: bool = False,
    camera_stereo: str = "left",
    orbit_params: Optional[dict] = None,
    gl_flip: bool = False,
):
    joints, gripper = load_joints_and_gripper(scene_dir)
    # gaussian_renderer lrs(..., pose_normalized=True) expects [-1,1]; DROID HDF5 is radians.
    joints = normalize_joints_from_chain(joints, gaussians)

    # Align DROID joints to the loaded DrRobot chain DOF. Different checkpoints
    # can expect different pose lengths (e.g. 9, 13, 15).
    lower, _upper = gaussians.chain.get_joint_limits()
    expected_dof = int(np.asarray(lower).shape[0])
    t_len = joints.shape[0]
    concatenated_joints = np.zeros((t_len, expected_dof), dtype=np.float32)
    used_dof = min(joints.shape[1], expected_dof)
    concatenated_joints[:, :used_dof] = joints[:, :used_dof].astype(np.float32, copy=False)
    print(
        f"[render_scene_video] chain_dof={expected_dof} source_dof={joints.shape[1]} "
        f"used={used_dof} -> pose shape {concatenated_joints.shape}"
    )

    cam_extr_traj: Optional[np.ndarray] = None
    cam_traj_path: Optional[str] = None
    if use_trajectory_camera:
        if camera_key in (None, "default"):
            raise ValueError("--trajectory_camera needs a real camera; use --camera_key wrist, ext1, or ext2.")
        md = load_scene_metadata(scene_dir)
        if md is None:
            raise FileNotFoundError(f"No metadata_*.json in {scene_dir}; cannot map camera_key to device serial.")
        serial = camera_serial_for_key(md, camera_key)
        cam_extr_traj, cam_traj_path = load_trajectory_camera_extrinsics(scene_dir, serial, camera_stereo)
        if cam_extr_traj.shape[0] != concatenated_joints.shape[0]:
            raise ValueError(
                f"Camera trajectory length {cam_extr_traj.shape[0]} != joint trajectory {concatenated_joints.shape[0]} "
                f"({cam_traj_path})"
            )
        print(
            f"[render_scene_video] viewpoint=trajectory {cam_traj_path} serial={serial} stereo={camera_stereo} "
            f"shape={cam_extr_traj.shape}"
        )

    camera = copy.deepcopy(base_camera)

    if orbit_params is not None:
        extrinsic = mujoco_orbit_extrinsic(**orbit_params)
        apply_extrinsic_to_camera(camera, extrinsic)
        print(f"[render_scene_video] viewpoint=orbit {orbit_params}")
    elif cam_extr_traj is None:
        extrinsic = parse_camera_extrinsic(scene_dir, camera_key, gl_flip=gl_flip)
        if extrinsic is not None:
            apply_extrinsic_to_camera(camera, extrinsic)
            print(
                f"[render_scene_video] viewpoint=static DROID (--camera_key={camera_key!r}). "
                "NOTE: DROID extrinsics are in the real-world robot-base frame which may not "
                "align with the MuJoCo scene frame used during training. If the view is wrong, "
                "use --azimuth/--elevation/--distance for a MuJoCo orbit camera instead."
            )
        elif camera_key == "default":
            print("[render_scene_video] viewpoint=default (training sample camera; no DROID extrinsic override).")
        else:
            print(
                "[render_scene_video] viewpoint=training sample only (no scene extrinsics for this camera_key); "
                "try wrist/ext1/ext2 or add camera_extrinsics.npy."
            )

    frames = []
    for t in range(concatenated_joints.shape[0]):
        if cam_extr_traj is not None:
            apply_extrinsic_to_camera(camera, droid_6dof_to_w2c(cam_extr_traj[t], gl_flip=gl_flip))
        camera.joint_pose = torch.from_numpy(concatenated_joints[t]).to(camera.data_device)
        frame = render(camera, gaussians, bg_color)["render"]
        frame = torch.clamp(frame, 0.0, 1.0)
        frame_np = (frame.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        frames.append(frame_np)

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    write_video(frames, output_path, fps)
    if gripper is not None:
        np.save(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_gripper.npy"), gripper)


def main():
    parser = argparse.ArgumentParser("Render all DROID scenes with DrRobot model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Directory from train.py (must contain cfg_args; use output/franka_emika_panda_complement1 in this repo)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/data/user_data/yjangir/yash/sidegig/wm/data/droid_10_demos",
        help="Path containing scene_* folders",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save rendered mp4 files")
    parser.add_argument(
        "--background",
        type=str,
        default="black",
        choices=["white", "black"],
        help="Rasterizer clear color behind Gaussians (was hardcoded white in load_model).",
    )
    def _fps_arg(value: str) -> Union[str, float]:
        """``match`` / ``auto`` = read FPS from DROID ``recordings/MP4/<serial>.mp4``; else positive float."""
        s = str(value).strip().lower()
        if s in ("match", "auto"):
            return "match"
        v = float(value)
        if v <= 0:
            raise argparse.ArgumentTypeError("--fps must be positive or match/auto")
        return v

    parser.add_argument(
        "--fps",
        type=_fps_arg,
        default="match",
        help="Output video frame rate. Default match: ffprobe the MP4 for --camera_key (e.g. 60). "
        "Use a number (e.g. 15) only for fixed-rate exports. Old default was 15, which made clips ~4x "
        "longer than real DROID 60 Hz video.",
    )
    parser.add_argument(
        "--camera_key",
        type=str,
        default="ext1",
        choices=["default", "wrist", "ext1", "ext2"],
        help="default = training sample camera only; wrist/ext1/ext2 = DROID metadata (and serial for --trajectory_camera).",
    )
    parser.add_argument(
        "--trajectory_camera",
        action="store_true",
        help="Use per-timestep extrinsics from trajectory.h5 (observation/camera_extrinsics/<serial>_<stereo>) "
        "instead of a single static pose from metadata.",
    )
    parser.add_argument(
        "--camera_stereo",
        type=str,
        default="left",
        choices=["left", "right"],
        help="ZED stereo eye when reading trajectory.h5 camera extrinsics (falls back to _left if _right is missing).",
    )
    parser.add_argument(
        "--azimuth", type=float, default=None,
        help="MuJoCo orbit camera azimuth (degrees). When set, overrides DROID extrinsics with "
        "a MuJoCo-convention orbit camera (same coordinate system the model was trained in). "
        "Requires --elevation and --distance too.",
    )
    parser.add_argument("--elevation", type=float, default=None, help="MuJoCo orbit camera elevation (degrees).")
    parser.add_argument("--distance", type=float, default=None, help="MuJoCo orbit camera distance from lookat.")
    parser.add_argument(
        "--lookat", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        help="MuJoCo orbit camera lookat point [x y z]. Default: origin.",
    )
    parser.add_argument(
        "--gl_flip", action="store_true",
        help="Apply 180° X-axis flip to DROID 6-DoF extrinsics (OpenCV→OpenGL). Off by default; "
        "for wm/data/droid_10_demos the no-flip path is correct. Enable only if the view is "
        "clearly upside-down or mirrored after checking --camera_key / --trajectory_camera.",
    )
    parser.add_argument("--width", type=int, default=None, help="Override render width (default: training resolution).")
    parser.add_argument("--height", type=int, default=None, help="Override render height (default: training resolution).")
    parser.add_argument(
        "--fov", type=float, default=None,
        help="Horizontal FOV in degrees (overrides per-camera ZED intrinsics). "
        "ZED 2i at 720p ≈ 100 deg. Default: auto from ZED intrinsics or training FOV.",
    )
    parser.add_argument("--max_scenes", type=int, default=-1, help="For debugging, limit number of scenes")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    orbit_flags = [args.azimuth is not None, args.elevation is not None, args.distance is not None]
    if any(orbit_flags) and not all(orbit_flags):
        parser.error("--azimuth, --elevation, and --distance must all be specified together.")

    orbit_params = None
    if args.azimuth is not None:
        orbit_params = dict(
            azimuth=args.azimuth, elevation=args.elevation,
            distance=args.distance, lookat=tuple(args.lookat),
        )

    safe_state(args.quiet)
    gaussians, bg_color, sample_cameras = load_model(args.model_path, background=args.background)
    base_camera = sample_cameras[0]

    render_w = args.width or (1280 if args.camera_key != "default" else None)
    render_h = args.height or (720 if args.camera_key != "default" else None)

    scene_dirs = sorted(glob.glob(os.path.join(args.dataset_root, "scene_*")))
    if args.max_scenes > 0:
        scene_dirs = scene_dirs[: args.max_scenes]

    if len(scene_dirs) == 0:
        raise RuntimeError(f"No scene directories found at {args.dataset_root}")

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        output_path = os.path.join(args.output_dir, f"{scene_name}.mp4")
        try:
            cam = copy.deepcopy(base_camera)

            # --- per-scene intrinsics (always returns a dict, falls back to defaults) ---
            zed = get_scene_camera_intrinsics(scene_dir, args.camera_key)
            zed_fx = zed["fx"]
            zed_fy = zed["fy"]
            print(f"[{scene_name}] ZED intrinsics: fx={zed_fx:.2f} fy={zed_fy:.2f}")

            w = render_w or base_camera.image_width
            h = render_h or base_camera.image_height
            override_camera_intrinsics(
                cam, w, h,
                fov_x_deg=args.fov, fx=zed_fx, fy=zed_fy,
            )
            print(
                f"[{scene_name}] render: {w}x{h}  "
                f"FoVx={np.degrees(cam.FoVx):.1f}°  FoVy={np.degrees(cam.FoVy):.1f}°"
            )

            if args.fps == "match":
                inferred = recording_fps_for_camera(scene_dir, args.camera_key)
                fps_use: Union[int, float] = inferred if inferred is not None else 15.0
                src = "recordings MP4" if inferred is not None else "fallback (no MP4 or default camera)"
                print(f"[{scene_name}] output fps={fps_use:.4g} ({src})")
            else:
                fps_use = float(args.fps)

            render_scene_video(
                scene_dir=scene_dir,
                output_path=output_path,
                gaussians=gaussians,
                bg_color=bg_color,
                base_camera=cam,
                fps=fps_use,
                camera_key=args.camera_key,
                use_trajectory_camera=args.trajectory_camera,
                camera_stereo=args.camera_stereo,
                orbit_params=orbit_params,
                gl_flip=args.gl_flip,
            )
            print(f"[OK] {scene_name} -> {output_path}")
        except Exception as e:
            print(f"[SKIP] {scene_name}: {e}")


if __name__ == "__main__":
    main()
