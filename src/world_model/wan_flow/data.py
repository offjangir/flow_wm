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
    idxs = np.linspace(0, len(all_frames) - 1, num_frames).astype(int)
    return [all_frames[i] for i in idxs]


def _load_tracks_npz(
    path: str, num_frames: int
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Load + temporally subsample sparse AllTracker tracks; normalize xy to [-1, 1].

    Returns:
        trajs:  ``(num_frames, N, 2)`` float32 in [-1, 1] (grid_sample coords)
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

    # Normalize pixel xy -> grid_sample [-1, 1].
    trajs_norm = trajs.copy()
    trajs_norm[..., 0] = (trajs_norm[..., 0] / max(W_track - 1, 1)) * 2.0 - 1.0
    trajs_norm[..., 1] = (trajs_norm[..., 1] / max(H_track - 1, 1)) * 2.0 - 1.0

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
        self.rows = df.to_dict("records")
        self.repeat = repeat

    def _abs(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.base_path, p)

    def assert_local_files_exist(self) -> None:
        """Fail fast if CSV points to missing videos / renders / tracks."""
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
                    self._abs(t), self.num_frames
                )
                sample["trajs"] = trajs           # (num_frames, N, 2) in [-1, 1]
                sample["visibs"] = visibs         # (num_frames, N) in {0, 1}
                sample["track_image_size"] = (H_t, W_t)

        return sample


def render_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch size 1; extend with padding before raising batch > 1."""
    if len(batch) != 1:
        raise NotImplementedError("Use batch_size=1 or implement padding in render_collate.")
    return batch[0]
