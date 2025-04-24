import time

import cv2
import matplotlib
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def rendered_depth(depth, cmap="Spectral_r"):
    cmap = matplotlib.colormaps.get_cmap(cmap)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    return (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def rectify_depth(pred: np.ndarray,
                  depth: np.ndarray = None,
                  max_depth: float = 5.0):
    if not isinstance(depth, np.ndarray): return pred
    assert depth.ndim == 2, f"Depth map should be 2D, but got {depth.ndim}D"
    mask = depth > 1 / max_depth
    x, y = pred[mask], depth[mask]
    # Linear least squares: Factorization
    xm, ym = x.mean(), y.mean()
    x_, y_ = x - xm, y - ym
    s = (x_ * y_).mean() / np.square(x_).mean()
    b = ym - s * xm
    pred = s * pred + b
    # for debug
    print(f"s={s}, b={b}, max={pred.max()}, RMSE={np.sqrt(np.square(pred[mask] - y).mean())}", )
    return pred


class DepthAnythingV2:
    """ https://github.com/DepthAnything/Depth-Anything-V2
        :param encoder: Encoder type (vits, vitb, vitl, vitg)"""

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
                 bgr: np.ndarray,
                 depth: np.ndarray = None,
                 max_depth: float = 5.0):
        # Affine-invariant inverse depth
        pred = self.model.infer_image(bgr, self.input_size)
        return rectify_depth(pred, depth, max_depth=max_depth)


if __name__ == '__main__':
    from utils.zjcv import Pinhole, RS_D435I
    from utils.realsense import rgbd_flow
    import open3d as o3d

    model = DepthAnythingV2("vitb")
    camera = Pinhole(**RS_D435I)

    if 0:
        # Infer a single image
        color = cv2.imread("assets/color.png")
        depth = cv2.imread("assets/depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth[depth > 0] = 5000 / depth[depth > 0]
        cv2.imshow("Depth", rendered_depth(depth))
        pred = model(color, depth)
        cv2.imshow("Pred", rendered_depth(pred))
        cv2.waitKey(0)

        # Depth to Point Cloud
        h, w = depth.shape[:2]
        pcd = camera.unproj(pred, color)
        o3d.visualization.draw_geometries([pcd])

    # Infer a video stream
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    view_control = vis.get_view_control()
    view_control.set_lookat([0, 0, -1])

    pcd = o3d.geometry.PointCloud()
    transform = np.eye(4)
    transform[1, 1] = transform[2, 2] = -1

    for color, depth in rgbd_flow(*camera.size):
        depth = depth.astype(np.float64)
        depth[depth > 0] = 1000 / depth[depth > 0]

        t0 = time.time()
        pred = model(color, depth, max_depth=3)
        fps = 1 / (time.time() - t0)
        print(f"FPS: {fps:.2f}")

        pcd = camera.unproj(pred, color, pcd=pcd, max_depth=3)
        pcd.transform(transform)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        # cv2.imshow("Depth", rendered_depth(pred)), cv2.waitKey(1)
    vis.destroy_window()