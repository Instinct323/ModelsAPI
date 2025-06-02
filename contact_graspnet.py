from collections import namedtuple
from pathlib import Path

import numpy as np

from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch.visualization_utils_o3d import draw_grasps


class GraspPoses(namedtuple("GraspPoses", ["T_grasp_cam", "scores", "contact_pts", "gripper_openings"])):

    def filter(self, mask):
        """ Filter grasps based on a boolean mask. """
        return GraspPoses(*(x[mask] for x in self))


class ContactGraspNet:
    """ Refer to contact_graspnet_pytorch/inference.py for more details.
        :param ckpt_dir: checkpoint directory """

    def __init__(self,
                 ckpt_dir: Path = Path("checkpoints/contact_graspnet")):
        self.estimator = GraspEstimator(config_utils.load_config(ckpt_dir, batch_size=1))
        CheckpointIO(checkpoint_dir=str(ckpt_dir / "checkpoints"), model=self.estimator.model).load('model.pt')

    def __call__(self,
                 pcd: np.ndarray):
        """
        :param pcd: point cloud with shape (N, 3)
        :returns: T_grasp_cam [n, 4, 4], scores [n,], contact_pts [n, 3], gripper_openings [n,]
        """
        return GraspPoses(*(x[-1] for x in self.estimator.predict_scene_grasps(pcd)))


if __name__ == '__main__':
    import open3d as o3d
    from utils.camera import Pinhole
    from utils.zjcv import to_colorful_pcd

    # Load RGB-D data
    struct = np.load("assets/rgb-depth-K.npy", allow_pickle=True).item()
    camera = Pinhole(img_size=struct["depth"].shape[::-1],
                     intrinsics=struct["K"][[0, 1, 0, 1], [0, 1, 2, 2]])
    pointmap = camera.pointmap(struct["depth"]).astype(np.float32)
    mask = (pointmap[..., 2] > 0) & (pointmap[..., 2] < 1.8)

    # Infer grasp poses
    cgn = ContactGraspNet()
    res = cgn(pointmap[mask])
    for i in res: print(i.shape)

    # Visualize point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(to_colorful_pcd(pointmap, struct["rgb"], mask))

    # Visualize grasps
    draw_grasps(vis, res[0], np.eye(4), res[-1])
    vis.run()
    vis.destroy_window()
