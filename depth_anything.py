import time

import cv2
import matplotlib
import numpy as np
import torch


def rendered_depth(depth, cmap="Spectral_r"):
    cmap = matplotlib.colormaps.get_cmap(cmap)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    return (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def rectify_depth(pred: np.ndarray,
                  depth: np.ndarray = None):
    if not isinstance(depth, np.ndarray): return pred
    assert depth.ndim == 2, f"Depth map should be 2D, but got {depth.ndim}D"
    mask = depth > 0
    x, y = pred[mask], depth[mask]
    # Linear least squares: Factorization
    xm, ym = x.mean(), y.mean()
    x_, y_ = x - xm, y - ym
    s = (x_ * y_).mean() / np.square(x_).mean()
    b = ym - s * xm
    return s * pred + b


class DepthAnythingV2:
    """ :param encoder: Encoder type (vits, vitb, vitl, vitg)"""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

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
        self.model = self.model.to(self.DEVICE).eval()

        self.input_size = input_size

    def __call__(self, bgr: np.ndarray, depth: np.ndarray = None):
        pred = self.model.infer_image(bgr, self.input_size)
        return rectify_depth(pred, depth)


if __name__ == '__main__':
    from utils import VideoCap

    model = DepthAnythingV2("vits")

    # Infer a single image
    color = cv2.imread("assets/color.png")
    depth = cv2.imread("assets/depth.png", cv2.IMREAD_UNCHANGED) / 5000
    cv2.imshow("Depth", rendered_depth(depth))
    cv2.imshow("Pred", rendered_depth(model(color, depth)))
    cv2.waitKey(0)

    # Infer a video stream
    srcs = VideoCap(0)
    for color in srcs:
        t0 = time.time()
        depth = model(color)
        fps = 1 / (time.time() - t0)
        print(f"FPS: {fps:.2f}")
        cv2.imshow("Depth", rendered_depth(depth))
        cv2.waitKey(1)
