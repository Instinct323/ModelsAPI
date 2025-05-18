from pathlib import Path

import numpy as np

from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator


class ContactGraspNet:
    """ Refer to contact_graspnet_pytorch/inference.py for more details.
        :param ckpt_dir: checkpoint directory """

    def __init__(self,
                 ckpt_dir: Path = Path("checkpoints/contact_graspnet")):
        self.estimator = GraspEstimator(config_utils.load_config(ckpt_dir, batch_size=1))
        CheckpointIO(checkpoint_dir=ckpt_dir / "checkpoints", model=self.estimator.model).load('model.pt')

    def __call__(self,
                 pcd: np.ndarray):
        """ :param pcd: point cloud with shape (N, 3)
            :returns: T_grasps_cam [n, 4, 4], scores [n,], contact_pts [n, 3], gripper_openings [n,]"""
        return [x[-1] for x in self.estimator.predict_scene_grasps(pcd)]


if __name__ == '__main__':
    cgn = ContactGraspNet()
    pcd = np.random.rand(20000, 3).astype(np.float32)
    res = cgn(pcd)
    for i in res:
        print(i.shape)
