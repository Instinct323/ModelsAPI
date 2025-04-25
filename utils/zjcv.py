from pathlib import Path
from typing import Union

import cv2
import numpy as np
import open3d as o3d
import matplotlib

RS_D435I = dict(w=640, h=480, fx=384.98394775390625, fy=384.98394775390625, cx=320.5026550292969, cy=240.8127899169922)
O3D_TRANSFORM = np.eye(4)
O3D_TRANSFORM[1, 1] = O3D_TRANSFORM[2, 2] = -1


def colormap(scale, cmap="Spectral_r", normalize=True):
    cmap = matplotlib.colormaps.get_cmap(cmap)
    if normalize: scale = (scale - scale.min()) / (scale.max() - scale.min())
    return (cmap(scale)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def to_colorful_pcd(points: np.ndarray,
                    mask: np.ndarray,
                    color: np.ndarray,
                    nb_points: int = 8,
                    radius: float = 0.02,
                    pcd: o3d.geometry.PointCloud = None):
    """ Convert depth map to point cloud"""
    pcd = pcd or o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color[mask][:, ::-1] / 255)
    pcd_, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pcd.points = pcd_.points
    pcd.colors = pcd_.colors
    return pcd


class Pinhole:

    def __init__(self, w, h, fx, fy, cx, cy):
        self.size = w, h
        self.intrinsic = np.array([fx, fy, cx, cy])
        # cache for unprojection
        self.__coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)
        self.__unproj = (self.__coords - [cx, cy]) / [fx, fy]

    def unproj(self,
               depth: np.ndarray,
               max_depth: float = 5):
        assert depth.ndim == 2, f"Depth map should be 2D, but got {depth.ndim}D"
        mask = (depth > 0) * (depth < max_depth)
        points = np.concatenate([self.__unproj[mask], depth[mask][..., None]], axis=-1)
        points[:, :2] *= points[:, -1:]
        return points, mask


class VideoCap(cv2.VideoCapture):
    """ 视频捕获
        :param src: 视频文件名称 (默认连接摄像头)
        :param delay: 视频帧的滞留时间 (ms)
        :param dpi: 相机分辨率"""

    def __init__(self,
                 src: Union[int, str, Path] = 0,
                 delay: int = 0,
                 dpi: list = None):
        src = str(src) if isinstance(src, Path) else src
        super().__init__(src)
        if not self.isOpened():
            raise RuntimeError("Failed to initialize video capture")
        self.delay = delay
        # 设置相机的分辨率
        if dpi:
            assert src == 0, "Only camera can set resolution"
            self.set(cv2.CAP_PROP_FRAME_WIDTH, dpi[0])
            self.set(cv2.CAP_PROP_FRAME_HEIGHT, dpi[1])

    def __iter__(self):
        def generator():
            while True:
                ok, image = self.read()
                if not ok: break
                if self.delay:
                    cv2.imshow("frame", image)
                    cv2.waitKey(self.delay)
                yield image
            # 回到开头
            self.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return generator()

    def __len__(self):
        return round(self.get(cv2.CAP_PROP_FRAME_COUNT))


if __name__ == '__main__':
    cam = Pinhole(80, 80, 100, 100, 40, 40)
