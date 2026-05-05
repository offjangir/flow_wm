import mujoco_py
import os

def render_scene(file_path, camera_position, joint_angles):
    # Load the model
    model = mujoco_py.load_model_from_path(file_path)
    sim = mujoco_py.MjSim(model)

    # Set camera position
    sim.data.cam.pos[0] = camera_position['x']
    sim.data.cam.pos[1] = camera_position['y']
    sim.data.cam.pos[2] = camera_position['z']

    # Set joint angles for the panda robot
    for joint_name, angle in joint_angles.items():
        joint_id = model.joint_name2id(joint_name)
        sim.data.qpos[joint_id] = angle

    # Create a viewer and render the scene
    viewer = mujoco_py.MjViewer(sim)
    while True:
        sim.step()
        viewer.render()

if __name__ == "__main__":
    # Path to the XML file containing the model
    file_path = 'path_to_your_panda_scene.xml'
    
    # Camera position (example)
    camera_position = {'x': 2.0, 'y': 2.0, 'z': 2.0}

    # Joint angles for the panda robot (example)
    joint_angles = {
        'joint1': 0.5,  # radians
        'joint2': 0.3,  # radians
        'joint3': -0.2  # radians
        # Add more joints if necessary
    }

    render_scene(file_path, camera_position, joint_angles)
