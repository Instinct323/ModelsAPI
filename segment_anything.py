import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def make_grid(w, h, s, axis=-1):
    arange = lambda n: np.arange(s / 2, n, s)
    return np.stack(np.meshgrid(*map(arange, (w, h))), axis=axis).astype(np.float32)


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores,
               point_coords=None, point_labels=None, box=None,
               borders=True, **kwargs):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert point_labels is not None
            show_points(point_coords, point_labels, plt.gca())
        if box is not None:
            # boxes
            show_box(box, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


class SegmentAnythingV2(SAM2ImagePredictor):
    """ https://github.com/facebookresearch/sam2
        :param encoder: Encoder type (tiny, small, base_plus, large)"""

    def __init__(self, encoder):
        encoder_type = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}
        assert encoder in encoder_type

        checkpoint = f"checkpoints/sam2.1_hiera_{encoder}.pt"
        model_cfg = f"configs/sam2.1/sam2.1_hiera_{encoder_type[encoder]}.yaml"
        super().__init__(build_sam2(model_cfg, checkpoint))

    def predict_and_show(self, image, **kwargs):
        ret = masks, scores, logits = self.predict(**kwargs)
        show_masks(image, masks, scores, **kwargs)
        return ret


if __name__ == '__main__':
    predictor = SegmentAnythingV2("large")
    image = cv2.imread("assets/color.png")
    point = np.array([310, 160])

    # TODO: batch inference
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        predictor.predict_and_show(image,
                                   point_coords=point[None],
                                   point_labels=np.ones(1, dtype=np.bool_))
