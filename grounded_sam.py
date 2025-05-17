import cv2
import numpy as np
import supervision as sv
import torch

from grounding_dino import GroundingDINO
from segment_anything import SegmentAnythingV2


class GroundedSAM:

    def __init__(self,
                 gdino_kwargs: dict = None,
                 sam2_kwargs: dict = None):
        self.gdino = GroundingDINO(**(gdino_kwargs or {}))
        self.gdino.anno_box.color_lookup = self.gdino.anno_label.color_lookup = sv.ColorLookup.INDEX
        self.sam2 = SegmentAnythingV2(**(sam2_kwargs or {}))

    def __call__(self,
                 image: np.ndarray,
                 classes: list[str]) -> sv.Detections:
        """ :param image: PIL image
            :param classes: list of classes to detect """
        # TODO: prompt 为 None, 使用 RAM 生成
        dets = self.gdino(image, classes)
        masks = self.sam2(image, box=dets.xyxy)
        dets.mask = masks["mask"]
        # Reorder by scores
        dets.confidence *= masks["confidence"]
        i = np.argsort(dets.confidence)[::-1]
        for k in ("xyxy", "confidence", "class_id", "mask"):
            setattr(dets, k, getattr(dets, k)[i])
        return dets

    def annotate(self,
                 image: np.ndarray,
                 detections: sv.Detections) -> np.ndarray:
        return self.sam2.annotate(self.gdino.annotate(image, detections=detections), masks=detections.mask)


if __name__ == '__main__':
    gsam = GroundedSAM(sam2_kwargs=dict(encoder="large"))
    image = cv2.imread("assets/color.png")

    with torch.inference_mode():
        dets = gsam(image, ["computer", "jar", "cup"])
        print(dets)
        cv2.imwrite("runs/gsam.png", gsam.annotate(image, dets))
