import warnings

import cv2
import numpy as np
import requests
import supervision as sv

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
                 box_thresh=0.35,
                 text_thresh=0.25,
                 nms_iou=0.5):
        # Check connection
        assert requests.get("https://huggingface.co", timeout=5).status_code == 200
        from groundingdino.config import GroundingDINO_SwinT_OGC as config
        checkpoint = "checkpoints/groundingdino_swint_ogc.pth"
        self.model = inference.Model(config.__file__, checkpoint)
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self.nms_iou = nms_iou

    def __call__(self,
                 image: np.ndarray,
                 classes: list[str]) -> sv.Detections:
        """ Open-vocabulary object detection
            :param image: PIL image
            :param classes: list of classes to detect """
        dets = self.model.predict_with_classes(
            image,
            classes=classes,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh
        )
        dets.metadata = np.array(classes)
        if self.nms_iou: dets = dets.with_nms(self.nms_iou)
        return dets

    def annotate(self,
                 image: np.ndarray,
                 detections: sv.Detections) -> np.ndarray:
        return self.anno_label.annotate(
            self.anno_box.annotate(image, detections=detections),
            detections=detections, labels=detection_labels(detections)
        )


if __name__ == "__main__":
    gdino = GroundingDINO()

    # Inference
    image = cv2.imread("assets/color.png")
    dets = gdino(image, ["computer", "jar", "cup"])
    print(dets)

    # Visualize
    cv2.imwrite("runs/gdino.jpg", gdino.annotate(image, dets))
