import cv2
import numpy as np
import supervision as sv
import torch
import torchvision

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils import sv_annotate


class SegmentAnythingV2:
    """ https://github.com/facebookresearch/sam2
        :param encoder: Encoder type (tiny, small, base_plus, large)"""
    anno_mask = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.7)
    anno_box = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    anno_label = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, smart_position=True)

    def __init__(self,
                 encoder: str,
                 mask_thresh: float = 0.,
                 points_per_side: int = 32):
        encoder_type = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}
        assert encoder in encoder_type

        checkpoint = f"checkpoints/sam2.1_hiera_{encoder}.pt"
        model_cfg = f"configs/sam2.1/sam2.1_hiera_{encoder_type[encoder]}.yaml"

        sam2 = build_sam2(model_cfg, checkpoint)
        self.predictor = SAM2ImagePredictor(sam2, mask_threshold=mask_thresh)
        # self.video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        self.auto_predictor = SAM2AutomaticMaskGenerator(sam2, mask_threshold=mask_thresh, points_per_side=points_per_side)

    def __call__(self,
                 image: np.ndarray,
                 **kwargs) -> sv.Detections:
        """ :param kwargs: use `prompt mode` if keyword parameters are provided, otherwise use `automatic mode` """
        if kwargs:
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(**kwargs, return_logits=False, multimask_output=False)
            if masks.ndim > 3:
                masks = masks.reshape(-1, *masks.shape[-2:])
                scores = scores.reshape(-1)
            return sv.Detections(xyxy=sv.mask_to_xyxy(masks), mask=masks.astype(np.bool_), confidence=scores)
        else:
            ret = self.auto_predictor.generate(image)
            return sv.Detections(
                xyxy=torchvision.ops.box_convert(torch.tensor([r["bbox"] for r in ret]), in_fmt="xywh", out_fmt="xyxy").numpy(),
                mask=np.stack([r["segmentation"] for r in ret]),
                confidence=np.array([r["predicted_iou"] for r in ret])
            )


SegmentAnythingV2.__call__.__doc__ = SAM2ImagePredictor.predict.__doc__

if __name__ == '__main__':
    from utils.realsense import rgbd_flow
    import matplotlib.pyplot as plt

    sam2 = SegmentAnythingV2("tiny")

    image = cv2.imread("assets/color.png")
    point = np.array([[310, 160]])

    # TODO: batch inference
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        dets = sam2(image,
                    point_coords=point,
                    point_labels=np.ones(len(point), dtype=np.bool_))
        print(dets)
        cv2.imwrite("runs/sam2.png", sv_annotate(image, dets, anno_box=False))

        dets = sam2(image)
        print(dets)
        cv2.imwrite("runs/sam2_all.png", sv_annotate(image, dets, anno_box=False))

    for c, d in rgbd_flow(640, 480, show=False):
        dets = sam2(c)
        print(dets.confidence)
        plt.imshow(sv_annotate(c, dets, anno_box=False))
        plt.pause(1e-3)
