import base64
import json
import logging
import os
import re
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


class JSONprompter(dict):

    def prompt(self):
        return ("\nYour response should be formatted as a JSON code block, without any additional text. "
                "The required fields are:\n" + "\n".join(f"- {k}: {v}" for k, v in self.items()))

    def decode(self,
               response: str):
        content = re.search(r"\{.*}", response, flags=re.S)
        try:
            return json.loads(content.group(0))
        except:
            raise ValueError(f"Invalid JSON response: {response}")


class GridAnnotator:

    def __init__(self,
                 ngrid: int,
                 color: tuple = (255, 255, 255),
                 thickness: float = 5e-3):
        self.ngrid = ngrid
        self.color = color
        self.thickness = thickness

    def make_grid(self, w, h):
        nr = round(np.sqrt(self.ngrid / h * w))
        nc = round(np.sqrt(self.ngrid * h / w))
        rows = np.round(np.linspace(0, h - 1, nr + 1)).astype(int)
        cols = np.round(np.linspace(0, w - 1, nc + 1)).astype(int)
        return dict(rows=rows, cols=cols, ngrid=nr * nc, shape=(nr, nc))

    def annotate(self,
                 image: np.ndarray,
                 grid_info: dict):
        image = image.copy()
        h, w = image.shape[:2]
        thickness = max(1, round(min(h, w) * self.thickness))
        for r in grid_info["rows"]: cv2.line(image, (0, r), (w, r), self.color, thickness)
        for c in grid_info["cols"]: cv2.line(image, (c, 0), (c, h), self.color, thickness)
        return image

    def index_grid(self,
                   grid_id: int,
                   grid_info: dict):
        if grid_id >= grid_info["ngrid"] or grid_id < 0: return None
        nr, nc = grid_info["shape"]
        r, c = grid_id // nc, grid_id % nc
        rows, cols = grid_info["rows"], grid_info["cols"]
        return np.array([cols[c], rows[r], cols[c + 1], rows[r + 1]])
