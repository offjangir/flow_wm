import mujoco
from pytorch_kinematics import chain, frame
from pytorch_kinematics.mjcf import _build_chain_recurse
import pytorch_kinematics.transforms as tf
from typing import Union


def build_chain_from_mjcf_path(path, body: Union[None, str, int] = None):
    """
    Build a Chain object from MJCF data.

    Parameters
    ----------
    path : str
        MJCF path
    body : str or int, optional
        The name or index of the body to use as the root of the chain. If None, body idx=0 is used.

    Returns
    -------
    chain.Chain
        Chain object created from MJCF.
    """
    m = mujoco.MjModel.from_xml_path(path)
    if body is None:
        root_body = m.body(0)
    else:
        root_body = m.body(body)
    root_frame = frame.Frame(root_body.name,
                             link=frame.Link(root_body.name,
                                             offset=tf.Transform3d(rot=root_body.quat, pos=root_body.pos)),
                             joint=frame.Joint())
    _build_chain_recurse(m, root_frame, root_body)
    return chain.Chain(root_frame)

if __name__ == "__main__":
    test_path = 'output/universal_robots_ur5e_experiment/robot_xml/scene.xml'
    chain = build_chain_from_mjcf_path(test_path)
    breakpoint()