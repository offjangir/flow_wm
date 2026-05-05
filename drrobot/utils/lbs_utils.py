# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from typing import NewType, Union, Optional
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch
import open3d as o3d
import numpy as np
import copy
import os
import sys
from tqdm import tqdm
import math
import torch.optim as optim
import glob
import traceback

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)

@dataclass
class ModelOutput:
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    v_shaped: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclass
class SMPLOutput(ModelOutput):
    betas: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None


@dataclass
class SMPLHOutput(SMPLOutput):
    left_hand_pose: Optional[Tensor] = None
    right_hand_pose: Optional[Tensor] = None
    transl: Optional[Tensor] = None


@dataclass
class SMPLXOutput(SMPLHOutput):
    expression: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None


@dataclass
class MANOOutput(ModelOutput):
    betas: Optional[Tensor] = None
    hand_pose: Optional[Tensor] = None


@dataclass
class FLAMEOutput(ModelOutput):
    betas: Optional[Tensor] = None
    expression: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None
    neck_pose: Optional[Tensor] = None


def find_joint_kin_chain(joint_id, kinematic_tree):
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


def to_tensor(
        array: Union[Array, Tensor], dtype=torch.float32
) -> Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

# pytorch dataset using above code
class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, path, N_points=3000):
        self.path = path
        self.N_points = N_points
        self.dirs = [d for d in glob.glob(os.path.join(self.path, 'sample_*')) if os.path.isdir(d)]
        print(f"[PointCloud] Found {len(self.dirs)} samples in {self.path}")
    
    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, idx):
        sample_path = self.dirs[idx]
        pcd = o3d.io.read_point_cloud(os.path.join(sample_path, 'pc.ply'))
        pcd_np = np.asarray(pcd.points)
        pcd = torch.tensor(pcd_np, dtype=torch.float32)
        
        # subsample to 5000 points
        try:
            idx = np.random.choice(pcd.shape[0], self.N_points, replace=False)
        except:
            print(f"population is {pcd.shape[0]} but sample size is {self.N_points}")
            idx = np.arange(pcd.shape[0])
        pcd = pcd[idx]

        pose = np.load(os.path.join(sample_path, 'joint_positions.npy'))
        pose = torch.tensor(pose, dtype=torch.float32)

        return pcd, pose