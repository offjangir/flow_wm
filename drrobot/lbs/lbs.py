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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from typing import Tuple, List
import numpy as np

import torch
import torch.nn.functional as F

from utils.lbs_utils import rot_mat_to_euler, Tensor
from pytorch_kinematics.chain import Chain

import torch
import rff
from copy import deepcopy
from utils.pt3d_utils import matrix_to_quaternion, quaternion_to_matrix
from torch.amp import autocast

positional_encoding = rff.layers.PositionalEncoding(sigma=1.0, m=5)
basic_encoding = rff.layers.BasicEncoding()


def find_dynamic_lmk_idx_and_bcoords(
    vertices: Tensor,
    pose: Tensor,
    dynamic_lmk_faces_idx: Tensor,
    dynamic_lmk_b_coords: Tensor,
    neck_kin_chain: List[int],
    pose2rot: bool = True,
) -> Tuple[Tensor, Tensor]:
    ''' Compute the faces, barycentric coordinates for the dynamic landmarks


        To do so, we first compute the rotation of the neck around the y-axis
        and then use a pre-computed look-up table to find the faces and the
        barycentric coordinates that will be used.

        Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
        for providing the original TensorFlow implementation and for the LUT.

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        pose: torch.tensor Bx(Jx3), dtype = torch.float32
            The current pose of the body model
        dynamic_lmk_faces_idx: torch.tensor L, dtype = torch.long
            The look-up table from neck rotation to faces
        dynamic_lmk_b_coords: torch.tensor Lx3, dtype = torch.float32
            The look-up table from neck rotation to barycentric coordinates
        neck_kin_chain: list
            A python list that contains the indices of the joints that form the
            kinematic chain of the neck.
        dtype: torch.dtype, optional

        Returns
        -------
        dyn_lmk_faces_idx: torch.tensor, dtype = torch.long
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
        dyn_lmk_b_coords: torch.tensor, dtype = torch.float32
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
    '''

    dtype = vertices.dtype
    batch_size = vertices.shape[0]

    if pose2rot:
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
    else:
        rot_mats = torch.index_select(
            pose.view(batch_size, -1, 3, 3), 1, neck_kin_chain)

    rel_rot_mat = torch.eye(
        3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).repeat(
            batch_size, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                    max=39)).to(dtype=torch.long)
    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals +
                   (1 - neg_mask) * y_rot_angle)

    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                           0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                          0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(
    vertices: Tensor,
    faces: Tensor,
    lmk_faces_idx: Tensor,
    lmk_bary_coords: Tensor
) -> Tensor:
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1).to(torch.long)).view(
        batch_size, -1, 3)
                        #The '.to(torch.long)'.
                        # added to make the trace work in c++,
                        # otherwise you get a runtime error in c++:
                        # 'index_select(): Expected dtype int32 or int64 for index'

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks

from functools import lru_cache

def batch_forward_kinematics(chain, pose):
    '''
    Differentiable forward kinematics with Pytorch Kinematics

    Parameters
    ----------
    chain: pytorch_kinematics.chain.Chain
        kinamtics chain object for performing differentiable forward kinematics
    pose: torch.tensor BxJ

    Returns
    -------
    A: torch.tensor BxJx4x4
        Affine transformation matrix of all joints
    '''
    with torch.amp.autocast('cuda', enabled=False):
        tg = chain.forward_kinematics(pose.float())
    joints = list(tg.keys())
    A = torch.stack([tg[joint]._matrix for joint in joints],dim=1)
    return A


def cache_decorator(cache_attr):
    def decorator(func):
        def wrapper(*args, **kwargs):
            global_vars = globals()
            if cache_attr not in global_vars:
                print(f"[Cache] {func.__name__}")
                global_vars[cache_attr] = func(*args, **kwargs)
            return global_vars[cache_attr]
        return wrapper
    return decorator

# @cache_decorator('_cached_nearest_W')
def get_nearest_W(dist, V):
    min_dist, min_idx = torch.min(dist, dim=-1)
    W = torch.zeros_like(dist)
    W[:, torch.arange(V), min_idx] = 1
    return W

# @cache_decorator('_cached_invdist_W')
def get_invdist_W(dist):
    epsilon = 1e-5
    inv_dist = 1.0 / (dist + epsilon)
    W = inv_dist / inv_dist.sum(dim=-1, keepdim=True)
    return W

@cache_decorator('_cached_canonical_pose')
def get_canonical_pose(lower_limits, upper_limits):
    #[N, 2]

    min_limits = toarray(lower_limits)
    max_limits = toarray(upper_limits)

    canonical_pose = np.zeros(lower_limits.shape[0])

    #nonzero joint indices are ones whree [min, max] does not overlap with 0
    nonzero_joint_indices = np.where((min_limits > 0) | (max_limits < 0))[0]
    # print(f"This robot has {len(nonzero_joint_indices)} nonzero joints, these are {nonzero_joint_indices}")
    
    #make nonzero joints as close to 0 as possible
    for i in nonzero_joint_indices:
        if min_limits[i] > 0:
            print(f"Setting joint {i} to {min_limits[i]}")
            canonical_pose[i] = min_limits[i]
        else:
            print(f"Setting joint {i} to {max_limits[i]}")
            canonical_pose[i] = max_limits[i]

    return torch.tensor(canonical_pose, dtype=torch.float32, device=lower_limits.device)

def toarray(tensor):
    return tensor.clone().detach().cpu().numpy()


def compute_and_cache_zero_pose_data(chain, v_template, lower_limits, upper_limits):
    """
    Returns
    -------
    A_zero_inverse: torch.tensor Nx(J + 1)x4x4
        The inverse of the zero pose transformation matrix
    distances: torch.tensor NxV
        The distances between joints and the vertices of zero pose
    offsets: torch.tensor NxVx(J + 1)x3
        The offsets from the zero pose
    """

    zero_pose = get_canonical_pose(lower_limits, upper_limits)
    A_zero = batch_forward_kinematics(chain, zero_pose)
    with autocast('cuda', enabled=False):
        A_zero_inverse = torch.linalg.inv(A_zero)

    zero_joint_positions = A_zero[:, :, :3, 3]

    offsets = zero_joint_positions[:, None] - v_template[:, :, None].detach()  # N, V, num_frames, 3
    distances = torch.norm(offsets, dim=-1)

    offsets = offsets[:, :, chain.joint_type_indices > 0, :]

    return (A_zero_inverse, distances, offsets)
    
def lrs(
    pose: Tensor,
    v_template: Tensor,
    lrs_weights: Tensor,
    chain: Chain,
    pose_normalized: bool = False,
    lrs_model = None,
    rotations = None,
) -> Tuple[Tensor, Tensor]:
    ''' Performs Linear Robot Skinning with the given shape and pose parameters

        Parameters
        ----------
        pose : torch.tensor Nx(J + 1)
            The pose parameters in axis-angle format
        v_template torch.tensor NxVx3
            The template mesh that will be deformed
        lrs_weights: torch.tensor V x (J + 1)
            The linear robot skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        chain: pytorch_kinematics.chain.Chain
            Differentiable forward kinematics

        Returns
        -------
        verts: torch.tensor NxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor NxJx3
            The joints of the model
    '''
    
    # import time
    # start_time = time.time()

    N, J = pose.shape
    V = v_template.shape[1]
    device, dtype = v_template.device, v_template.dtype
    num_joints = pose.shape[1]
    num_frames = len(chain.joint_type_indices)

    # unnormalized pose back to uppwer and lower limits
    lower_limits, upper_limits = chain.get_joint_limits()
    lower_limits = torch.tensor(lower_limits, dtype=dtype, device=device)
    upper_limits = torch.tensor(upper_limits, dtype=dtype, device=device)

    if pose_normalized:
        pose = (pose + 1) * (upper_limits - lower_limits) / 2 + lower_limits

    A = batch_forward_kinematics(chain, pose)
    
    with autocast('cuda', enabled=False):
        A_zero_inverse, dist, offsets = compute_and_cache_zero_pose_data(chain, v_template, lower_limits, upper_limits)
    A = A @ A_zero_inverse

    if lrs_model is not None: # implicit LRS
        v_template_unencoded = v_template.reshape(N * V, 3)
        v_template_encoded = positional_encoding(v_template_unencoded)
        # reshaped_offsets = offsets.reshape(N * V, J * 3)

        reshaped_offsets = offsets.reshape(N * V, J, 3)
        unencoded_offsets = reshaped_offsets.reshape(N * V, J * 3)
        encoded_offsets = positional_encoding(reshaped_offsets).reshape(N*V, -1)

        distances = torch.norm(offsets, dim=-1)
        unencoded_distances = distances.reshape(N * V, J)
        encoded_distances = positional_encoding(distances).reshape(N * V, -1)

        network_input = torch.cat([v_template_encoded, encoded_offsets, encoded_distances], dim=1)
        network_input = torch.cat([network_input, unencoded_offsets, v_template_unencoded, unencoded_distances], dim=1)

        # network_input = torch.cat([v_template_encoded, encoded_offsets], dim=1)
        W = lrs_model(network_input)
        W = W.reshape(N, V, num_frames)
    else:
        # W is N x V x (J + 1)
        residual_W = lrs_weights.unsqueeze(dim=0).expand([N, -1, -1])
        raise NotImplementedError("This is not implemented yet")

    #Dr. Robot: In our initial experiments, we used this prior
    # W = alpha * get_nearest_W(dist, V) + (1-alpha) * get_invdist_W(dist)
    # # W = get_invdist_W(dist)
    # W = W / W.sum(dim=-1, keepdim=True)
    # # print(residual_W.shape)
    # W = W + residual_W

    # hack_temperature = 2
    W = torch.softmax(W, dim=-1)

    # print("W shape after softmax", W.shape)

    #print list with .3f precision
    # print("wieghts per joint", [f"{item:.3f}" for item in W.mean(dim=(0, 1)).clone().detach().cpu().numpy().tolist()])
    
    
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    T = torch.matmul(W, A.view(N, num_frames, 16)) \
        .view(N, -1, 4, 4)
    # T is N x V x 4 x 4

    homogen_coord = torch.ones([N, v_template.shape[1], 1],
                               dtype=dtype, device=device)
    v_template_homo = torch.cat([v_template, homogen_coord], dim=2)

    v_homo = torch.matmul(T, torch.unsqueeze(v_template_homo, dim=-1))

    if rotations is not None:
        nonhomogen_rotation_matrices = quaternion_to_matrix(rotations) #V, 3, 3
        #make it 4x4 homogeneous
        homogen_rotation_matrices = torch.cat([
            torch.cat([nonhomogen_rotation_matrices, v_template[0, :, :, None]], dim=2),
            torch.cat([torch.zeros(V, 1, 3, device=rotations.device), torch.ones(V, 1, 1, device=rotations.device)], dim=2)
        ], dim=1)

        transformed_rotation_matrices = T @ homogen_rotation_matrices

        transformed_rotation_matrices_3by3 = transformed_rotation_matrices[:, ..., :3, :3]
        # breakpoint()

        transformed_rotations = matrix_to_quaternion(transformed_rotation_matrices_3by3)
    
        return transformed_rotation_matrices[:, :, :3, 3], transformed_rotations
    else:
        return v_homo[:, :, :3, 0], None

def deform(pose, pc_template, pc_deformed, deform_model):
    '''
    Calculate delta changes applied to scale, rotation, opacity, shs conditioned
    on pose, initial pointcloud and deformed pointcloud by lrs model

    Input:
    pose: N x J
    pc_template: N x V x 3
    pc_defomred: N x V x 3

    Return:
    delta_scale: N x V x 3
    delta_rotation: N x V x 4
    delta_opacity: N x V x 1
    delta_shs: N x V x 16 x 3
    '''

    N, V, _ = pc_template.shape
    J = pose.shape[-1]
    # positional encode both pointclouds and concatenate
    pc_templated_encoded = positional_encoding(pc_template.reshape(N * V, 3))
    pc_deformed_encoded = positional_encoding(pc_deformed.reshape(N * V, 3))
    pose = pose.reshape(N * V, J) # expand pose for each vertices
    input_features = torch.cat([pose, pc_templated_encoded, pc_deformed_encoded], dim=1)
    output = deform_model(input_features)
    delta_scale, delta_rot, delta_opacity, delta_shs = output[:, :3], output[:, 3:7], output[:, 7:8], output[:, 8:]
    # delta_rot = F.normalize(delta_rot, p=2, dim=1)  # normalize quaternion
    delta_shs = delta_shs.view(N, V, 16, 3)  # reshape to N x V x 16 x 3
    return delta_scale, delta_rot, delta_opacity, delta_shs

def pose_conditioned_deform(
    pc_template, pc_deformed, scales, rotations, opacity, shs, joints, deform_model
):
    '''
    Calculate delta changes applied to scale, rotation, opacity, shs conditioned
    on pose, initial pointcloud and deformed pointcloud by lrs model

    Input:
    pc_template: N x V x 3
    pc_defomred: N x V x 3
    scales: N x V x 3
    rotations: N x V x 4
    opacity: N x V x 1
    shs: N x V x 16 x 3
    joints: N x J x 3

    Return:
    scales: N x V x 3
    rotations: N x V x 4
    opacity: N x V x 1
    shs: N x V x 16 x 3
    '''
    pose = joints
    delta_scale, delta_rot, delta_opacity, delta_shs = deform(pose, pc_template, pc_deformed, deform_model)
    
    scales = scales + delta_scale 
    rotations = rotations + delta_rot
    opacity = opacity + delta_opacity 
    shs = shs + delta_shs
    return scales, rotations, opacity, shs

def lbs(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    pose2rot: bool = True,
    test: bool = False,
) -> Tuple[Tensor, Tensor]:
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def vertices2joints(J_regressor: Tensor, vertices: Tensor) -> Tensor:
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas: Tensor, shape_disps: Tensor) -> Tensor:
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32
) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms