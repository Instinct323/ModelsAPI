import matplotlib
import numpy as np
import open3d as o3d

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
               depth: np.ndarray):
        assert depth.ndim == 2, f"Depth map should be 2D, but got {depth.ndim}D"
        pcd = np.repeat(depth[..., None], 3, axis=-1)
        pcd[..., :2] *= self.__unproj
        return pcd


if __name__ == '__main__':
    cam = Pinhole(80, 80, 100, 100, 40, 40)
    d = np.random.rand(80, 80).astype(np.float32) * 10
    print(cam.unproj(d).shape)
