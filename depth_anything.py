import time

import cv2
import numpy as np
import torch

from utils.zjcv import *

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def rectify_depth(pred: np.ndarray,
                  depth: np.ndarray,
                  mask: np.ndarray,
                  show_res: int = None):
    """
    Rectify the predicted depth map using the ground truth depth map.
    :param pred: Predicted inverse depth map.
    :param depth: Ground truth depth map.
    :param mask: Mask for valid depth values.
    :param show_res: Show the rectified result for debugging.
    """
    assert depth.ndim == 2, f"Depth map should be 2D, but got {depth.ndim}D"
    x, y = pred[mask], 1 / depth[mask]
    # Linear least squares: Factorization
    xm, ym = x.mean(), y.mean()
    x_, y_ = x - xm, y - ym
    s = (x_ * y_).mean() / np.square(x_).mean()
    b = ym - s * xm
    pred = 1 / (s * pred + b)
    # for debug
    if show_res is not None:
        print(f"s={s}, b={b}, max={pred.max()}, RMSE={np.sqrt(np.square(1 / pred[mask] - y).mean())}")
        to_show = np.concatenate(list(map(colormap, [pred, np.abs(pred - depth)])), axis=1)
        cv2.imshow("Pred & Res", to_show)
        cv2.waitKey(show_res)
    return pred


class DepthAnythingV2:
    """
    https://github.com/DepthAnything/Depth-Anything-V2
    :param encoder: Encoder type (vits, vitb, vitl, vitg)
    """

    def __init__(self, encoder, input_size=518):
        from depth_anything_v2 import dpt

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.model = dpt.DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = self.model.to(DEVICE).eval()

        self.input_size = input_size

    def __call__(self,
                 bgr: np.ndarray):
        """
        Infer depth from a BGR image.
        :param bgr: BGR image with shape (H, W, 3) and dtype uint8
        :return: Affine-invariant inverse depth map with shape (H, W) and dtype float32
        """
        return self.model.infer_image(bgr, self.input_size)


if __name__ == '__main__':
    import open3d as o3d
    from utils.camera import Camera

    camera = Camera.from_yaml("config/RS-D435i.yaml")
    model = DepthAnythingV2("vitb")


    def depth_mask(depth):
        return (depth > 0) * (depth < 5.0)


    if 1:
        # Infer a single image
        color = cv2.imread("assets/desktop-c.png")
        depth = cv2.imread("assets/desktop-d.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000

        # Depth to Point Cloud
        t0 = time.time()
        pred = rectify_depth(model(color), depth, depth_mask(depth))
        print("FPS:", 1 / (time.time() - t0))
        pcd = remove_radius_outlier(to_colorful_pcd(camera.pointmap(pred), color, depth_mask(pred)))
        # pcd.transform(O3D_TRANSFORM)

        transform = np.eye(4)
        transform[0, 3] = 3
        org = remove_radius_outlier(to_colorful_pcd(camera.pointmap(depth), color, depth_mask(depth)))
        org.transform(transform)
        o3d.visualization.draw_geometries([pcd, org, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)])

    # Infer a video stream
    from utils.realsense import rgbd_flow

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    for color, depth in rgbd_flow(*camera.img_size):
        depth = depth.astype(np.float64) / 1000

        t0 = time.time()
        pred = rectify_depth(model(color), depth, depth_mask(mask))
        fps = 1 / (time.time() - t0)
        print(f"FPS: {fps:.2f}")

        mask = depth_mask(pred)
        pcd = remove_radius_outlier(to_colorful_pcd(camera.pointmap(pred), color, mask, pcd=pcd))
        pcd.transform(O3D_TRANSFORM)

        vis.add_geometry(pcd, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()
