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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.robot_dataset import RobotDataset

import numpy as np
from scene.gaussian_model import BasicPointCloud
from plyfile import PlyData, PlyElement
from typing import NamedTuple
import pytorch_kinematics as pk
from utils.pk_utils import build_chain_from_mjcf_path
import shutil
import pickle

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = None
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

#Dr. Robot scene class
class RobotScene:
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, 
                 gaussians : GaussianModel, 
                 load_iteration=None, 
                 opt_params=None,
                 from_ckpt=False,
                 n_sample_cameras=64):
        """b
        :param path: Path to colmap scene main folder.

        n_sample_cameras: number of sample cameras, which is used for visualization/validation/

        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.opt_params = opt_params
        self.n_sample_cameras = n_sample_cameras
        
        self.canonical_cameras=None
        self.pose_conditioned_cameras=None
        self.sample_cameras=None

        assert load_iteration or opt_params.dataset_path, "Either load_iteration or dataset_path must be provided"

        if load_iteration:
            #if you're loading from a previous iteration, we assume that you're not continuing training and just loading the model
            self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud")) \
                if load_iteration == -1 else load_iteration
            print("[Scene] Loading trained Gaussian model at iteration {}".format(self.loaded_iter))

            #load sample cameras from pkl file
            with open(os.path.join(self.model_path, "sample_cameras.pkl"), 'rb') as f:
                self.sample_pose_conditioned_cameras = pickle.load(f)
        else:
            print("[Scene] Loading datasets for training")
            #if you're not loading from a previous iteration, load the datasets for training
            self.load_datasets()
            
            #save self.sample_cameras as a pkl file
            with open(os.path.join(self.model_path, "sample_cameras.pkl"), 'wb') as f:
                pickle.dump(self.sample_pose_conditioned_cameras, f)


        self.dataset_type = None
        self.cameras_extent = 2.0 #TODO don't hardcode this
        self.args = args

        if opt_params.k_plane: #TODO add this back
            print("[Scene] Initializing k-plane deformation network with {} joints.".format(self.num_joints))

            self.gaussians.initialize_deformation_network(self.num_joints) 
            xyz_max = np.array([1.0, 1.0, 1.0])
            xyz_min = np.array([-1.0, -1.0, -1.0])
            self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)

        #During dataset generation, for convenience purposes, the model xml was copied to the dataset directory
        #When training a new model, we also copy the xml to the directory where checkpoints are saved. This makes it so that one only needs the model path.
        if not os.path.exists(os.path.join(self.model_path, "robot_xml/scene.xml")):
            assert os.path.exists(os.path.join(opt_params.dataset_path, "robot_xml/scene.xml")), "Robot XML file not found in dataset, something is wrong"
            print("[Scene] Copying robot model into model path")
            shutil.copytree(os.path.join(opt_params.dataset_path, "robot_xml"), os.path.join(self.model_path, "robot_xml"))
        else:
            print("[Scene] Loading robot model from model path")

        # breakpoint()
       
        chain = build_chain_from_mjcf_path(os.path.join(args.model_path, "robot_xml/scene.xml"))
        num_frames = len(chain.joint_type_indices) #might be different from num_joints, includes world frame too
        self.num_joints = chain.n_joints
        self.gaussians.initialize_lrs_model(self.num_joints, num_frames)
        self.gaussians.initialize_kinematic_chain(chain)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "pose_conditioned_iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(self.point_cloud, self.cameras_extent, self.num_joints)

    def load_datasets(self):
        self.canonical_cameras = RobotDataset(dataset_path=self.opt_params.dataset_path, 
                                       num_init_gaussians=self.opt_params.num_init_gaussians,
                                       stage='canonical')
        self.pose_conditioned_cameras = RobotDataset(dataset_path=self.opt_params.dataset_path, 
                                       num_init_gaussians=self.opt_params.num_init_gaussians,
                                       stage='pose_conditioned')
        self.sample_canonical_cameras = [self.canonical_cameras[i] for i in range(min(self.n_sample_cameras, len(self.canonical_cameras)))]
        self.sample_pose_conditioned_cameras = [self.pose_conditioned_cameras[i] for i in range(min(self.n_sample_cameras, len(self.pose_conditioned_cameras)))]
        self.point_cloud = fetchPly(self.canonical_cameras.plypath)

    def save(self, iteration, stage):
        if stage == "canonical":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/canonical_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/pose_conditioned_iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, stage='canonical'):
        if stage == 'canonical':
            return self.canonical_cameras
        else:
            return self.pose_conditioned_cameras

    def getSampleCameras(self, stage='canonical'):
        if stage == 'canonical':
            return self.sample_canonical_cameras
        else:
            return self.sample_pose_conditioned_cameras

#Original 3D Gaussians implementation of scene
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]