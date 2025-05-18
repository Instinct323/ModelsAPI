import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch

from grounding_dino import GroundingDINO
from segment_anything import SegmentAnythingV2


class GroundedSAM:

    def __init__(self,
                 score_thresh: float = 0.1,
                 gdino_kwargs: dict = None,
                 sam2_kwargs: dict = None,
                 tag2text_kwargs: dict = None):
        self.score_thresh = score_thresh
        self.gdino = GroundingDINO(**(gdino_kwargs or {}), tag2text_kwargs=tag2text_kwargs)
        self.gdino.anno_box.color_lookup = self.gdino.anno_label.color_lookup = sv.ColorLookup.INDEX
        self.sam2 = SegmentAnythingV2(**(sam2_kwargs or {}))

    def __call__(self,
                 image: np.ndarray,
                 caption: str = None) -> sv.Detections:
        """ :param image: PIL image
            :param caption: text prompt """
        # TODO: prompt 为 None, 使用 RAM 生成
        dets = self.gdino(image, caption)
        if dets:
            masks = self.sam2(image, box=dets.xyxy)
            dets.mask = masks["mask"]
            # Reorder by scores
            dets.confidence *= masks["confidence"]
            dets = dets[np.argsort(dets.confidence)[::-1]]
        if self.score_thresh:
            dets = dets[dets.confidence > self.score_thresh]
        return dets

    def annotate(self,
                 image: np.ndarray,
                 detections: sv.Detections) -> np.ndarray:
        ret = self.gdino.annotate(image, detections=detections)
        if detections.mask is not None: ret = self.sam2.annotate(ret, masks=detections.mask)
        return ret


if __name__ == '__main__':
    from utils.realsense import rgbd_flow

    gsam = GroundedSAM(sam2_kwargs=dict(encoder="large"), tag2text_kwargs={})

    image = cv2.imread("assets/color.png")
    dets = gsam(image)
    print(dets)
    cv2.imwrite("runs/gsam.png", gsam.annotate(image, dets))

    with torch.inference_mode():
        for c, d in rgbd_flow(640, 480, show=False):
            dets = gsam(c)
            print(dets.confidence)
            plt.imshow(gsam.annotate(c, dets))
            plt.pause(1e-3)
