import cv2
import numpy as np
import supervision as sv
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SegmentAnythingV2:
    """ https://github.com/facebookresearch/sam2
        :param encoder: Encoder type (tiny, small, base_plus, large)"""
    anno_mask = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.7)

    def __init__(self, encoder):
        encoder_type = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}
        assert encoder in encoder_type

        checkpoint = f"checkpoints/sam2.1_hiera_{encoder}.pt"
        model_cfg = f"configs/sam2.1/sam2.1_hiera_{encoder_type[encoder]}.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        # self.video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    def __call__(self, image, **kwargs) -> dict:
        """ :returns: dict["mask", "confidence"]"""
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(**kwargs, return_logits=False, multimask_output=False)
        if masks.ndim > 3:
            masks = masks.reshape(-1, *masks.shape[-2:])
            scores = scores.reshape(-1)
        return dict(mask=masks.astype(np.bool_), confidence=scores)

    def annotate(self,
                 image: np.ndarray,
                 masks: np.ndarray) -> np.ndarray:
        return self.anno_mask.annotate(image.copy(), detections=sv.Detections(xyxy=np.zeros([len(masks), 4]), mask=masks))


if __name__ == '__main__':
    sam2 = SegmentAnythingV2("large")

    image = cv2.imread("assets/color.png")
    point = np.array([[310, 160]])

    # TODO: batch inference
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = sam2(image,
                     point_coords=point,
                     point_labels=np.ones(len(point), dtype=np.bool_))
        print(masks)
        cv2.imwrite("runs/sam2.png", sam2.annotate(image, masks["mask"]))
