import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import glob
import torch
from tqdm import tqdm
import time
from scene.cameras import Camera
import open3d as o3d
import pickle
import torchvision

class RobotDataset(Dataset):
    def __init__(
        self,
        dataset_path=None,
        stage='canonical',
        blend_n_point_clouds=10,
        num_init_gaussians=5_000,
    ):
        """
        Args:
            robot_dataset_params (RobotDatasetParams): optional parameters for the dataset
            stage (str): 'coarse' (canonical gaussian training), 'fine' (pose-conditioned training)
            blend_n_point_clouds (int): number of point clouds to blend for the initialization of the 3D gaussian
            each point cloud comes from a sample in the data, and while all point clouds are expected to be of the same size, 
            blending many of them initially helps to sample more uniformly.
        """
        super().__init__()
        assert stage in ['canonical', 'pose_conditioned']
        self.dataset_path = dataset_path
        self.plypath = os.path.join(self.dataset_path, "points3D_multipleview.ply")
        self.robot_metadata = pickle.load(open(os.path.join(self.dataset_path, "robot_metadata.pkl"), "rb"))

        self.joint_limits = self.robot_metadata['joint_limits']
        self.n_joints = len(self.joint_limits)
        self.iters = 0


        print(f"[RobotDataset] Joint limits for {self.dataset_path}:")
        for key, value in self.robot_metadata['joint_range_dict'].items():
            print(f"{key}: {value}")
        # Calculate number of trainable joints based on configuration

        canonical_samples = glob.glob(os.path.join(self.dataset_path, "canonical_sample_*"))
        samples = glob.glob(os.path.join(self.dataset_path, "sample_*"))

        assert len(canonical_samples) > 0, f"No canonical samples found in {self.dataset_path}"
        assert len(samples) > 0, f"No samples found in {self.dataset_path}"

        first_canonical_sample_dirs = canonical_samples[:blend_n_point_clouds]
        pcs = o3d.geometry.PointCloud()
        for first_canonical_sample_dir in first_canonical_sample_dirs:
            pc_path = os.path.join(first_canonical_sample_dir, "pc.ply")
            pcs += o3d.io.read_point_cloud(pc_path)
        print(f"[RobotDataset] Total of points first {blend_n_point_clouds} in canonical samples:", len(pcs.points))

        voxel_size = 0.0001
        while len(pcs.points) > num_init_gaussians:
            voxel_size *= 1.01
            pcs = pcs.voxel_down_sample(voxel_size=voxel_size)
        print("[RobotDataset] Pointcloud downsampled to:", len(pcs.points))  
        o3d.io.write_point_cloud(self.plypath, pcs)    


        if stage == 'canonical':
            self.data_dirs = [d for d in glob.glob(os.path.join(self.dataset_path, "canonical_sample_*")) if os.path.isdir(d)]
        else:
            self.data_dirs = [d for d in glob.glob(os.path.join(self.dataset_path, "sample_*")) if os.path.isdir(d)]

        assert len(self.data_dirs) > 0, f"No {'canonical ' if stage == 'canonical' else ''}sample directories found in {self.dataset_path}"

    def normalize_joint_positions(self, joint_positions):

        #normalize between [-1, 1]
        lower_limits = self.joint_limits[:, 0]
        upper_limits = self.joint_limits[:, 1]
        scale = 2 / (upper_limits - lower_limits)
        normalized_joint_positions = (joint_positions - lower_limits) * scale - 1.

        assert lower_limits.shape == joint_positions.shape, f"{lower_limits.shape} != {joint_positions.shape}"
        assert joint_positions.shape == normalized_joint_positions.shape, f"{joint_positions.shape} != {normalized_joint_positions.shape}"

        assert np.abs(normalized_joint_positions).max() <= 1.0 + 1e-6
      
        return normalized_joint_positions

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):

        #seed a generator with index and load that image
        generator = np.random.RandomState(self.iters)
        idx = generator.choice(len(self.data_dirs))
        self.iters += 1

        data_dir = self.data_dirs[idx]
        try:
            images = glob.glob(os.path.join(data_dir, "image_*.jpg"))
            images = sorted(images, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            extrinsics_matrices = np.load(os.path.join(data_dir, "extrinsics.npy"))
            raw_joint_params = np.load(os.path.join(data_dir, "joint_positions.npy"))
        except Exception as e:
            print(f"Error {e} loading data, skipping...")
            return self.__getitem__(idx + 1)
        
        img_idx = generator.choice(len(images))

        image_path = os.path.join(data_dir, "image_{}.jpg".format(img_idx))
        depth_path = os.path.join(data_dir, "depth_{}.npy".format(img_idx)) 
        
        #load image as numpy array and put it between 0,1
        try:
            img = np.array(Image.open(image_path)).astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            depth = np.load(depth_path)

            extrinsic_matrix = extrinsics_matrices[img_idx]
            joint_params = self.normalize_joint_positions(raw_joint_params)
        except Exception as e:
            print(f"Error {e} loading data, skipping...")
            return self.__getitem__(idx + 1)
        
        FovX, FovY = 0.78, 0.78 #default fov from mujoco, TODO: implement it in a more principled way

            
        return Camera(
            colmap_id=None,
            R=extrinsic_matrix[:3, :3].T,
            T=extrinsic_matrix[:3, 3],
            FoVx=FovX,
            FoVy=FovY,
            image=img,
            gt_alpha_mask=None,
            image_name=None,
            uid=None,
            joint_pose=joint_params,
            depth=depth
        )
 

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt


    dataset = RobotDataset(dataset_path="data/universal_robots_ur5e", stage='pose_conditioned')

    start = time.time()
    n = 0
    for camera in dataset:
        print("looping!")
        n += 1
        print("avg time:", (time.time() - start) / n)
