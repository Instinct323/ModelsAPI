import warnings

import cv2
import numpy as np
import requests
import supervision as sv
from PIL import Image

warnings.filterwarnings("ignore")

from groundingdino.util import inference


def detection_labels(detections: sv.Detections):
    return [
        f"{phrase} {score:.2f}"
        for phrase, score
        in zip(detections.metadata[detections.class_id], detections.confidence)
    ]


class GroundingDINO:
    anno_box = sv.BoxAnnotator()
    anno_label = sv.LabelAnnotator(smart_position=True)

    def __init__(self,
                 box_thresh: float = 0.35,
                 text_thresh: float = 0.25,
                 nms_iou: float = 0.5,
                 tag2text_kwargs: dict = None):
        # Check connection
        assert requests.get("https://huggingface.co", timeout=5).status_code == 200
        from groundingdino.config import GroundingDINO_SwinT_OGC as config
        self.model = inference.Model(config.__file__, "checkpoints/groundingdino_swint_ogc.pth")
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self.nms_iou = nms_iou
        # Recognize Anything
        self.tag2text = None
        if tag2text_kwargs is not None:
            from tag2text import Tag2Text
            self.tag2text = Tag2Text(**tag2text_kwargs)

    def __call__(self,
                 image: np.ndarray,
                 caption: str = None) -> sv.Detections:
        """ Open-vocabulary object detection
            :param image: PIL image
            :param caption: text prompt """
        if not caption:
            assert self.tag2text is not None, "Please provide caption or use tag2text_kwargs to enable Tag2Text."
            caption = self.tag2text(Image.fromarray(image))[0].replace(" |", ".")
            print("Tag2Text:", caption)

        dets, phrases = self.model.predict_with_caption(
            image, caption,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh
        )
        dets.metadata = classes = np.array(sorted(set(phrases)))
        dets.class_id = np.searchsorted(classes, phrases)
        if self.nms_iou: dets = dets.with_nms(self.nms_iou)
        return dets

    def annotate(self,
                 image: np.ndarray,
                 detections: sv.Detections) -> np.ndarray:
        return self.anno_label.annotate(
            self.anno_box.annotate(image.copy(), detections=detections),
            detections=detections, labels=detection_labels(detections)
        )


if __name__ == "__main__":
    gdino = GroundingDINO(tag2text_kwargs={})

    image = cv2.imread("assets/color.png")

    dets = gdino(image, "computer. jar. cup")
    print(dets)
    cv2.imwrite("runs/gdino_wo_tt.jpg", gdino.annotate(image, dets))

    dets = gdino(image)
    print(dets)
    cv2.imwrite("runs/gdino_w_tt.jpg", gdino.annotate(image, dets))
