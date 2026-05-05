#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from lbs.lbs import lrs, pose_conditioned_deform

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage='canonical'):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    assert stage in ['canonical', 'pose_conditioned']
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug if pipe is not None else False,
        # compute_grad_cov2d=True,
        # proj_k=viewpoint_camera.projection_matrix
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    #if joints is not tensor, make it tensor
    if viewpoint_camera.joint_pose is not None and not isinstance(viewpoint_camera.joint_pose, torch.Tensor):
        joints = torch.from_numpy(viewpoint_camera.joint_pose).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        joints = viewpoint_camera.joint_pose.to(means3D.device).repeat(means3D.shape[0],1)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    
    if stage == 'pose_conditioned':
        if pc.args.k_plane: # using k-plane deformation model for deformation instead of linear blend skinning
            means3D, scales, rotations, opacity, shs = pc._deformation(means3D, scales, rotations, opacity, shs, times_sel=None, joints=joints)
            rotations = pc.rotation_activation(rotations)
        else: #use lbs for means+rotations, + MLP for appearance
            means3D_deformed, rotations_deformed = lrs(joints[0][None].float(), means3D[None], pc._lrs, pc.chain, pose_normalized=True, lrs_model=pc.lrs_model, rotations=rotations)
            
            if pc.args.no_appearance_deformation: # without learnable deformation model on scales, opacity, and spherical harmonics
                scales_out, _rotations_out, opacity_out, shs_out = scales[None], rotations[None], opacity[None], shs[None]
            else:
                scales_out, _rotations_out, opacity_out, shs_out = \
                    pose_conditioned_deform(means3D[None], means3D_deformed[None], scales[None], rotations[None], \
                                            opacity[None], shs[None], joints[None].float(), pc.appearance_deformation_model)
            
            #these are calculated using joint transformations
            means3D = means3D_deformed[0]
            rotations = rotations_deformed[0]
            
            #these are learned during lrs
            scales = scales_out[0]
            opacity = opacity_out[0]
            shs = shs_out[0]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # breakpoint()
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}
