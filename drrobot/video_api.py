import torch
from scene import Scene, RobotScene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render

def initialize_gaussians(model_path=None, from_ckpt=False):
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    args = get_combined_args(parser)

    if model_path is not None:
        args.model_path = model_path


    gaussians = GaussianModel(model.sh_degree, opt)
    
    scene = RobotScene(args, gaussians, opt_params=opt, from_ckpt=True, load_iteration=-1)

    gaussians.model_path = scene.model_path #todo, find a cleaner way to do this

    bg_color = [1, 1, 1]
    background_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    return gaussians, background_color, scene.getSampleCameras(stage='pose_conditioned'), gaussians.chain


if __name__ == "__main__":
    import moviepy as mpy
    from moviepy.editor import ImageSequenceClip
    import copy
    import numpy as np
    from PIL import Image

    #trajectory values are between [-1, 1] and they represent [-pi, pi] joint angles
    joint_trajectory_begin = torch.tensor([-0.0, -0.0, -0.0, 0.0, 0.0, 0.0])
    joint_trajectory_end = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    #interpolate t=20
    T = 20
    joint_trajectory = torch.stack([joint_trajectory_begin + (joint_trajectory_end - joint_trajectory_begin) * t / T for t in range(T)])

    gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians()

    for i, camera in enumerate(sample_cameras):
        my_camera = camera
        break

    video = []
    for i in range(len(joint_trajectory)):
        my_camera.joint_pose = joint_trajectory[i]
        frame = render(my_camera, gaussians, background_color)['render'] #3, 480, 480
        video.append(frame)
        
        # frame_pil = Image.fromarray((frame.clone().detach() * 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        # frame_pil.save(f"out_img.png")

    video = torch.stack(video)
    video = torch.clamp(video, 0, 1)
    video = video.cpu().detach().numpy()
    video = np.transpose(video, (0, 2, 3, 1))    
    video = (video * 255).astype(np.uint8)

    clip = ImageSequenceClip(list(video), fps=10)
    clip.write_videofile("out_video.mp4")

