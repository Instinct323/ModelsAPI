import matplotlib
import numpy as np
import open3d as o3d

O3D_TRANSFORM = np.eye(4)
O3D_TRANSFORM[1, 1] = O3D_TRANSFORM[2, 2] = -1


def colormap(scale, cmap="Spectral_r", normalize=True):
    cmap = matplotlib.colormaps.get_cmap(cmap)
    if normalize: scale = (scale - scale.min()) / (scale.max() - scale.min())
    return (cmap(scale)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def to_colorful_pcd(pointmap: np.ndarray,
                    color: np.ndarray,
                    mask: np.ndarray,
                    pcd: o3d.geometry.PointCloud = None):
    """
    :param pointmap: 2D dense field of points [H, W, 3]
    :param color: RGB color [H, W, 3]
    :param mask: mask of the points [H, W]
    """
    pcd = pcd or o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointmap[mask])
    pcd.colors = o3d.utility.Vector3dVector(color[mask][:, ::-1] / 255)
    return pcd


def remove_radius_outlier(pcd: o3d.geometry.PointCloud,
                          nb_points: int = 8,
                          radius: float = 0.02):
    pcd_, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pcd.points = pcd_.points
    pcd.colors = pcd_.colors
    return pcd


if __name__ == '__main__':
    pass
