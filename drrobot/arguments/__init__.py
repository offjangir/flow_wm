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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = False

        super().__init__(parser, "Loading Parameters", sentinel)


    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 10000
        self.densify_from_iter = 500
        self.densify_until_iter = 5_000
        self.densify_grad_threshold = 0.0002
        self.random_background = True

        self.dataset_path = "" #this should be the path that generate_robot_data.py was run from
        self.num_init_gaussians = 10_000 #initial number of points to use for the gaussian model

        ####### kplane params, not relevant to dr. robot  ############
        self.k_plane = False
        self.net_width = 64
        self.timebase_pe = 4
        self.defor_depth = 1
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6
        self.plane_tv_weight = 0.0001
        self.time_smoothness_weight = 0.01
        self.l1_time_planes = 0.0001
        self.kplanes_config = { #these are for xyz, this gets edited depending on n_joints
            'grid_dimensions': 2,
            'input_coordinate_dim': 3,
            'output_coordinate_dim': 8,
            'resolution': [64, 64, 64]
        }
        self.multires = [1, 2, 4, 8]
        self.no_dx=False
        self.no_grid=False
        self.no_ds=False
        self.no_dr=False
        self.no_do=True
        self.no_dshs=True
        self.empty_voxel=False
        self.grid_pe=0
        self.static_mlp=False
        self.apply_rotation=False
        self.concat_plane_features=False
        self.gaussian_initialization = False
        self.do_heuristic_rotations=False
        self.do_train_rotations=False 
        self.do_train_scales_shs=False
        ###################################


        #### lrs + dr.robot params ####
        self.learn_rotations = False
        self.lambda_depth = 0.5
        self.canonical_training_iterations = 2_000
        self.pose_conditioned_training_iterations = 15_000

        self.lrs_train_steps = 1e6 #minimum of this and the number of samples in dataset
        self.lrs_train_epochs = 1
        self.lrs_lr = 1e-4

        self.appearance_deform_lr = 1e-5
        self.no_appearance_deformation = True

        ####################

        self.gaussian_training_batch_size = 8

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
