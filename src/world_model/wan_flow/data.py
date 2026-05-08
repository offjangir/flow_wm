"""
Training data: CSV metadata with video (real camera), caption, render, tracks.

CSV columns:

- ``video``:  path to the real camera ``.mp4`` (target — what the DiT denoises).
- ``prompt``: text caption.
- ``render``: path to the DrRobot render ``.mp4`` (per-timestep action conditioning).
- ``tracks`` (optional): path to AllTracker sparse-track ``.npz`` extracted
  from the *same* real camera video. Provides ``trajs (T_track, N, 2)``,
  ``visibs (T_track, N)``, and ``image_size [H, W]`` so we can normalize the
  pixel coordinates to ``[-1, 1]`` for the auxiliary :class:`TracksHead` loss.

Paths are resolved relative to ``base_path`` unless absolute.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_video_frames(path: str, num_frames: int) -> List[Image.Image]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(path, "ffmpeg")
    try:
        all_frames = [Image.fromarray(fr).convert("RGB") for fr in reader]
    finally:
        reader.close()
    if len(all_frames) == 0:
        raise ValueError(f"No frames read from {path}")
    if len(all_frames) < num_frames:
        # Silent frame duplication via np.linspace makes short clips look like they
        # trained fine but the model only ever sees ``len(all_frames)`` distinct
        # frames per sample. Fail fast so the caller picks shorter num_frames.
        raise ValueError(
            f"{path}: only {len(all_frames)} frames available, but num_frames={num_frames} "
            f"was requested. Either lower num_frames or drop this clip from the manifest."
        )
    idxs = np.linspace(0, len(all_frames) - 1, num_frames).astype(int)
    return [all_frames[i] for i in idxs]


def _load_actions_npz(
    path: str,
    num_frames: int,
    stats: Optional[Dict[str, np.ndarray]] = None,
    field: str = "state",
) -> np.ndarray:
    """Load (T_raw, D) action stream and subsample to ``num_frames`` indices
    that match the camera-frame loader.

    DROID convention: the camera mp4 has ``T_raw - 1`` frames (one trailing
    "post-final" action step is recorded after the last camera capture).
    Truncating to the first ``T_raw - 1`` action steps yields a 1:1 index
    correspondence with camera frames; ``np.linspace`` with the same rule as
    :func:`_load_video_frames` then picks the same indices.

    Args:
        path:       absolute path to ``.npz`` written by ``scripts/extract_actions.py``.
        num_frames: temporal length to return (matches the video loader).
        stats:      optional ``{"mean": (D,), "std": (D,)}`` for z-score
                    normalization. Pass ``None`` for raw values.
        field:      which array in the npz to read (``state`` for 8-d
                    joint+gripper; ``action`` for 7-d cmd target+gripper).

    Returns:
        ``(num_frames, D) float32`` numpy array.
    """
    blob = np.load(path)
    if field not in blob.files:
        raise KeyError(
            f"{path}: missing field {field!r}; available={list(blob.files)}"
        )
    arr = np.asarray(blob[field], dtype=np.float32)            # (T_raw, D)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D, got shape {arr.shape}")
    T_raw = arr.shape[0]
    if T_raw < 2:
        raise ValueError(f"{path}: too few timesteps ({T_raw})")
    # Drop trailing post-final step → align with camera-frame indexing.
    n_video_frames = T_raw - 1
    if n_video_frames < num_frames:
        raise ValueError(
            f"{path}: only {n_video_frames} usable action steps but "
            f"num_frames={num_frames} requested."
        )
    idxs = np.linspace(0, n_video_frames - 1, num_frames).astype(int)
    sub = arr[idxs]                                            # (num_frames, D)
    if stats is not None:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        sub = (sub - mean) / std
    return sub.astype(np.float32, copy=False)


def _load_action_stats(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not path:
        return None
    with open(path, "r") as f:
        blob = json.load(f)
    return {
        "mean": np.asarray(blob["mean"], dtype=np.float32),
        "std": np.asarray(blob["std"], dtype=np.float32),
    }


def _load_tracks_npz(
    path: str,
    num_frames: int,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Load + temporally subsample sparse AllTracker tracks; normalize xy to [-1, 1].

    Tracks were extracted at AllTracker's working canvas ``(H_track, W_track)``
    (preserves the source video's aspect ratio). The DiT trains on a fixed
    ``(target_height, target_width)`` resize that may NOT preserve aspect
    ratio (DROID 1280×720 → 240×432 changes ~16:9 → 9:5). If we normalize
    using ``(W_track, H_track)`` the tracks land in the AllTracker canvas
    coords; once the video is squished to the training canvas, the tracks
    are off the gripper. Re-project pixel coords into the training canvas
    *before* normalizing whenever both dimensions are provided.

    Returns:
        trajs:  ``(num_frames, N, 2)`` float32 in [-1, 1] (grid_sample coords,
            in the training canvas if target_h/w supplied, else AllTracker canvas)
        visibs: ``(num_frames, N)`` float32 in {0, 1}
        (H_track, W_track): the AllTracker working frame size, useful for
            converting back to pixels at viz time.
    """
    z = np.load(path)
    trajs = z["trajs"].astype(np.float32)             # (T, N, 2) pixel xy
    visibs = z["visibs"].astype(np.float32)           # (T, N)
    H_track, W_track = (int(z["image_size"][0]), int(z["image_size"][1]))
    T_track = trajs.shape[0]

    # Subsample to num_frames matching the video frames the DiT trains on.
    idxs = np.linspace(0, T_track - 1, num_frames).astype(int)
    trajs = trajs[idxs]                                # (num_frames, N, 2)
    visibs = visibs[idxs]                              # (num_frames, N)

    if target_height is not None and target_width is not None:
        sx = float(target_width) / float(max(W_track, 1))
        sy = float(target_height) / float(max(H_track, 1))
        denom_w = max(target_width - 1, 1)
        denom_h = max(target_height - 1, 1)
    else:
        sx = sy = 1.0
        denom_w = max(W_track - 1, 1)
        denom_h = max(H_track - 1, 1)

    trajs_norm = trajs.copy()
    trajs_norm[..., 0] = (trajs_norm[..., 0] * sx / denom_w) * 2.0 - 1.0
    trajs_norm[..., 1] = (trajs_norm[..., 1] * sy / denom_h) * 2.0 - 1.0

    return (
        torch.from_numpy(trajs_norm),
        torch.from_numpy(visibs),
        (H_track, W_track),
    )


class RenderI2VMetadataDataset(Dataset):
    """
    CSV columns: ``video``, ``prompt``, ``render``, optional ``tracks``.

    - ``video``:  path to the real camera ``.mp4`` (target).
    - ``prompt``: text caption.
    - ``render``: path to the DrRobot render ``.mp4`` (conditioning).
    - ``tracks``: path to AllTracker sparse tracks ``.npz`` (auxiliary
      supervision). When present, ``__getitem__`` returns
      ``trajs (T, N, 2) in [-1, 1]`` and ``visibs (T, N) in {0, 1}``.

    Both videos are evenly subsampled to ``num_frames`` so they align
    temporally after VAE encoding; tracks are subsampled the same way.
    """

    def __init__(
        self,
        base_path: str,
        metadata_csv: str,
        num_frames: int,
        height: int,
        width: int,
        repeat: int = 1,
        action_stats_path: Optional[str] = None,
        action_field: str = "state",
    ):
        self.base_path = os.path.abspath(base_path)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        df = pd.read_csv(metadata_csv)
        required = {"video", "prompt", "render"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"metadata CSV missing columns: {missing}")
        # Empty CSV prompt fields read back as NaN by default; normalize them
        # here so training encodes the empty prompt, not the literal string "nan".
        df["prompt"] = df["prompt"].fillna("").astype(str)
        self.has_tracks = "tracks" in df.columns
        # Actions present iff the column exists AND every row has a non-empty
        # path. Mixed batches (some rows with, some without) would force the
        # model into "actions=None" branch on those samples and silently train
        # an action-aware adapter without supervision; treat as all-or-nothing.
        if "actions" in df.columns:
            df["actions"] = df["actions"].fillna("").astype(str)
            n_have = (df["actions"].str.len() > 0).sum()
            self.has_actions = bool(n_have == len(df))
            if 0 < n_have < len(df):
                raise ValueError(
                    f"metadata CSV {metadata_csv}: 'actions' present in "
                    f"{n_have}/{len(df)} rows. Either populate every row or "
                    f"drop the rows with empty 'actions'."
                )
        else:
            self.has_actions = False
        self.action_stats = _load_action_stats(action_stats_path)
        self._action_stats_path = action_stats_path or ""
        self.action_field = action_field
        self.rows = df.to_dict("records")
        self.repeat = repeat

    def _abs(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.base_path, p)

    def assert_local_files_exist(self) -> None:
        """Fail fast if CSV points to missing videos / renders / tracks / actions."""
        missing: List[str] = []
        for i, row in enumerate(self.rows):
            for key in ("video", "render"):
                p = self._abs(row[key])
                if not os.path.isfile(p):
                    missing.append(f"  row {i + 1} column {key!r}: {p}")
            if self.has_tracks:
                t = row.get("tracks", "") or ""
                if t:
                    pt = self._abs(t)
                    if not os.path.isfile(pt):
                        missing.append(f"  row {i + 1} column 'tracks': {pt}")
            if self.has_actions:
                a = row.get("actions", "") or ""
                if a:
                    pa = self._abs(a)
                    if not os.path.isfile(pa):
                        missing.append(f"  row {i + 1} column 'actions': {pa}")
        if not missing:
            return
        head = "\n".join(missing[:25])
        tail = f"\n  ... ({len(missing) - 25} more)" if len(missing) > 25 else ""
        raise FileNotFoundError(
            f"{len(missing)} paths from metadata are not existing files.\n"
            f"dataset_base_path (resolved) = {self.base_path}\n"
            f"{head}{tail}\n"
            "Place .mp4 / .npz files at those paths, or point --dataset_base_path / "
            "--metadata_csv at a folder that contains them."
        )

    def __len__(self) -> int:
        return len(self.rows) * self.repeat

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx % len(self.rows)]
        vpath = self._abs(row["video"])
        rpath = self._abs(row["render"])

        video = _load_video_frames(vpath, self.num_frames)
        render = _load_video_frames(rpath, self.num_frames)

        sample: Dict[str, Any] = {
            "video": video,
            "render": render,
            "prompt": str(row["prompt"]),
        }

        if self.has_tracks:
            t = row.get("tracks", "") or ""
            if t:
                trajs, visibs, (H_t, W_t) = _load_tracks_npz(
                    self._abs(t), self.num_frames,
                    target_height=self.height, target_width=self.width,
                )
                sample["trajs"] = trajs           # (num_frames, N, 2) in [-1, 1]
                sample["visibs"] = visibs         # (num_frames, N) in {0, 1}
                sample["track_image_size"] = (H_t, W_t)

        if self.has_actions:
            a = row.get("actions", "") or ""
            if a:
                sample["actions"] = _load_actions_npz(
                    self._abs(a),
                    self.num_frames,
                    stats=self.action_stats,
                    field=self.action_field,
                )                                  # (num_frames, D) float32

        return sample


def render_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch size 1; extend with padding before raising batch > 1."""
    if len(batch) != 1:
        raise NotImplementedError("Use batch_size=1 or implement padding in render_collate.")
    return batch[0]
