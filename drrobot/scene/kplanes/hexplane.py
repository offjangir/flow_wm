import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    #grid shape [1, 256, 64, 64]
    #coords shape [4417, 2]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp


def get_coo_combs(in_dim, grid_nd):
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    #only keep combs that have 0,1,2 in them
    coo_combs = [comb for comb in coo_combs if (0 in comb) or (1 in comb) or (2 in comb)]
    return coo_combs


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        coo_combs: List[Tuple[int, int]],
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5,
        gaussian_initialization: bool = False
        ):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim >= 4
    assert grid_nd <= in_dim

    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))

        # if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
        #     nn.init.ones_(new_grid_coef)
        # if has_time_planes and np.arange(3, in_dim) in coo_comb:  # Initialize time planes to 1
        if gaussian_initialization:
            #Initialize with gaussian weights
            # nn.init.normal_(new_grid_coef, mean=0.0, std=0.1) 
            nn.init.normal_(new_grid_coef, mean=0.0, std=1.0)
        else:
            #this is how k-planes initializes by default, kinda crazy if you ask me
            if has_time_planes and (coo_comb[0] >= 3 or coo_comb[1] >= 3) :  # Initialize time planes to 1
                print("time planes for coo comb:", coo_comb)
                assert len(coo_comb) == 2, "only combs of 2 for now (alper)"
                nn.init.ones_(new_grid_coef)
            else:
                nn.init.uniform_(new_grid_coef, a=a, b=b)

        grid_coefs.append(new_grid_coef)

    return grid_coefs



def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            coo_combs: List[Tuple[int, int]],
                            num_levels: Optional[int],
                            concat_plane_features: bool = True
                            ) -> torch.Tensor:
    # coo_combs = list(itertools.combinations(
    #     range(pts.shape[-1]), grid_dimensions)
    # )

    # print("[interolate]: concatenating plane features:", concat_plane_features)
    # print("[interpolate] Number of combinations: ", len(coo_combs))
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = [] if concat_plane_features else 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = grid_sample_wrapper(grid[ci], pts[..., coo_comb]) #[1, 256, 64, 64], [4417, 2]
            # print("[interpolate] Interpolated plane shape 1:", interp_out_plane.shape)
            interp_out_plane = interp_out_plane.view(-1, feature_dim) 
            # print("[interpolate] Interpolated plane shape 2:", interp_out_plane.shape)
            # compute product over planes
            if concat_plane_features:
                interp_space.append(interp_out_plane)
            else:
                interp_space = interp_space * interp_out_plane

        if concat_plane_features:
            interp_space = torch.cat(interp_space, dim=-1)


        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        # print("[interpolate] Concatenating features after", multi_scale_interp.shape)

    # print("density mean:",torch.mean(multi_scale_interp).item())
    # print("density std:",torch.std(multi_scale_interp).item())
    # print("max density:",torch.max(multi_scale_interp).item())

    # print("[interpolate] multi scale interp shape:", multi_scale_interp.shape)
    return multi_scale_interp


class HexPlaneField(nn.Module):
    def __init__(
        self,
        bounds,
        planeconfig,
        multires,
        concat_plane_features=False,
        gaussian_initialization=False,
        args=None,
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        self.concat_plane_features = concat_plane_features

        # 1. Init planes
        self.coo_combs = get_coo_combs(in_dim=self.grid_config[0]["input_coordinate_dim"], 
                                        grid_nd=self.grid_config[0]["grid_dimensions"])
        print("coo_combs:",self.coo_combs)


        self.spatial_indices = np.arange(3)
        self.temporal_indices = np.arange(3, self.grid_config[0]["input_coordinate_dim"])

        #spatial if only has 0,1,2 in it
        self.is_spatial_comb = lambda comb: all([c in [0, 1, 2] for c in comb])
        #temporal if only has >=3 in it
        self.is_temporal_comb = lambda comb: all([c >= 3 for c in comb])
        #spatiotemporal if has one of each
        self.is_spatiotemporal_comb = lambda comb: any([c in [0, 1, 2] for c in comb]) and any([c >= 3 for c in comb])

        self.spatial_combs = []
        self.spatial_comb_indices = []
        self.temporal_combs = []
        self.temporal_comb_indices = []
        self.spatiotemporal_combs = []
        self.spatiotemporal_comb_indices = []

        # Iterate through each combination and its index
        for ci, comb in enumerate(self.coo_combs):
            if self.is_spatial_comb(comb):
                self.spatial_combs.append(comb)
                self.spatial_comb_indices.append(ci)
            if self.is_temporal_comb(comb):
                self.temporal_combs.append(comb)
                self.temporal_comb_indices.append(ci)
            if self.is_spatiotemporal_comb(comb):
                self.spatiotemporal_combs.append(comb)
                self.spatiotemporal_comb_indices.append(ci)

        assert len(self.spatial_comb_indices) + len(self.temporal_comb_indices) + len(self.spatiotemporal_comb_indices) == len(self.coo_combs), \
            "Spatial, temporal and spatiotemporal combs must cover all coo_combs"
        

        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                coo_combs=self.coo_combs,
                reso=config["resolution"],
                gaussian_initialization=args.gaussian_initialization
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self,xyz_max, xyz_min):


        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)


        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)
        print("Size of voxel plane:",torch.abs(self.aabb[0] - self.aabb[1]))

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None, joints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb)
        # print("timestamps shape:",timestamps.shape)
        # print("pts shape:",pts.shape)
        # print("joints shape:",joints.shape)
        # print("joints example",joints[0])
        pts = torch.cat((pts, joints), dim=-1).float()  # [n_rays, n_samples, 4]

        assert pts.shape[1] == len(self.grid_config[0]['resolution']), \
            f"Input points have {pts.shape[1]} dimensions, expected {len(self.grid_config[0]['resolution'])}"

        #if there's a point greater than 1, and it's in the first 3 dimensions

        #Print nan indices of points if there's any nan in it
        if torch.isnan(pts).any():
            print("nan indices:",torch.isnan(pts).nonzero(as_tuple=True))
            breakpoint()

        # print("pts shape:",pts.shape)
        assert pts[:, 3:].min() >= -1.00 and pts[:, 3:].max() <= 1.00, f"Input points must be in range [-1, 1], got {pts[:, 3:].min()} to {pts[:, 3:].max()}"

        #if there's a point in the first 3 that's out of range, then warn with red text
        if pts[:, :3].min() < -1.0 or pts[:, :3].max() > 1.0:
            print("\033[91m" + "Warning: Input points must be in range [-1, 1]" + "\033[0m", f"got {pts[:, :3].min()} to {pts[:, :3].max()}")

        pts = pts.reshape(-1, pts.shape[-1])

        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, 
            coo_combs=self.coo_combs,
            num_levels=None, concat_plane_features=self.concat_plane_features)
        
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)


        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                joints: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        features = self.get_density(pts, timestamps, joints)

        return features
