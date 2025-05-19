import warnings

import cv2
import numpy as np
import supervision as sv

from utils import detection_labels

warnings.filterwarnings("ignore")

from groundingdino.util import inference


class GroundingDINO:
    """ Reference: https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/automatic_label_tag2text_demo.py"""
    anno_box = sv.BoxAnnotator()
    anno_label = sv.LabelAnnotator(smart_position=True)

    def __init__(self,
                 box_thresh: float = 0.35,
                 text_thresh: float = 0.25,
                 nms_iou: float = 0.5,
                 boxarea_thresh: float = 0.7):
        # Require to connect to Huggingface
        # from groundingdino.config import GroundingDINO_SwinT_OGC as config
        self.model = inference.Model("checkpoints/GroundingDINO_SwinT_OGC.py", "checkpoints/groundingdino_swint_ogc.pth")
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self.nms_iou = nms_iou
        self.boxarea_thresh = boxarea_thresh

    def __call__(self,
                 image: np.ndarray,
                 caption: str) -> sv.Detections:
        """ Open-vocabulary object detection
            :param image: PIL image
            :param caption: text prompt """
        dets, phrases = self.model.predict_with_caption(
            image, caption,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh
        )
        dets.metadata = classes = np.array(sorted(set(phrases)))
        dets.class_id = np.searchsorted(classes, phrases)
        if self.boxarea_thresh:
            dets = dets[dets.box_area / np.prod(image.shape[:2]) < self.boxarea_thresh]
        if self.nms_iou: dets = dets.with_nms(self.nms_iou)
        return dets

    @classmethod
    def annotate(cls,
                 image: np.ndarray,
                 detections: sv.Detections) -> np.ndarray:
        return cls.anno_label.annotate(
            cls.anno_box.annotate(image.copy(), detections=detections),
            detections=detections, labels=detection_labels(detections)
        )


if __name__ == "__main__":
    gdino = GroundingDINO()

    image = cv2.imread("assets/color.png")

    dets = gdino(image, "computer. jar. cup")
    print(dets)
    cv2.imwrite("runs/gdino.jpg", gdino.annotate(image, dets))
