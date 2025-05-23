import base64
import logging
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import supervision as sv

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("utils")


def huggingface_model_path(repo_id: str) -> Path:
    """ Download the model from Hugging Face. """
    path = Path(f"~/.cache/huggingface/hub/models--{repo_id.replace('/', '--')}/snapshots").expanduser()
    if not path.exists():
        print("Downloading model from Hugging Face...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.system(f"huggingface-cli download {repo_id}")
    return next(path.iterdir())


def sv_annotate(image: np.ndarray,
                detections: sv.Detections,
                mask_opacity: float = 0.7,
                anno_box: bool = True,
                smart_label: bool = True) -> np.ndarray:
    """ :param image: OpenCV image
        :param detections: Supervision Detections with xyxy, confidence ..."""
    color_lookup = sv.ColorLookup.CLASS if detections.mask is None else sv.ColorLookup.INDEX
    image = image.copy()

    if anno_box:
        anno_box = sv.BoxAnnotator(color_lookup=color_lookup)
        image = anno_box.annotate(image, detections=detections)

    if detections.mask is not None:
        anno_mask = sv.MaskAnnotator(color_lookup=color_lookup, opacity=mask_opacity)
        image = anno_mask.annotate(image, detections=detections)

    if detections.confidence is not None:
        anno_label = sv.LabelAnnotator(color_lookup=color_lookup, smart_position=smart_label)
        labels = [f"{score:.2f}" for score in detections.confidence] if detections.class_id is None else [
            f"{phrase} {score:.2f}" for phrase, score in zip(detections.metadata[detections.class_id], detections.confidence)]
        image = anno_label.annotate(image, detections=detections, labels=labels)

    return image


class ContentMaker:

    def __init__(self,
                 local_run: bool,
                 max_img_size: int = 640):
        self.local_run = local_run
        self.max_img_size = max_img_size

    def process_text(self, text: str):
        assert isinstance(text, str)
        return text

    def process_image(self, img: np.ndarray):
        assert isinstance(img, np.ndarray)
        if max(img.shape[:2]) > self.max_img_size:
            img = sv.resize_image(img, (self.max_img_size,) * 2, keep_aspect_ratio=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return sv.cv2_to_pillow(img) if self.local_run else (
                f"data:image/jpg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1]).decode("utf-8"))

    def __call__(self,
                 role: str,
                 text: str,
                 image: Union[np.ndarray, list[np.ndarray]] = None):
        """ :param role: user, assistant, system """
        to_element = lambda t, v: {"type": t, t: v}
        ret = [to_element("text", self.process_text(text))]
        if image is not None:
            # Multiple images are supported
            img_t = "image" if self.local_run else "image_url"
            for img in [image] if isinstance(image, np.ndarray) else image:
                ret.append(to_element(img_t, self.process_image(img)))
        return {"role": role, "content": ret}
