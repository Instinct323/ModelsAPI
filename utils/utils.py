import logging
import os
from pathlib import Path

import PIL.Image
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


def make_content(role, **contents) -> dict:
    """ :param role: user, assistant, system
        :param contents: The input contents (e.g., text, image, video)."""
    ret = []
    for t, v in contents.items():
        if t in ("image", "video"):
            if isinstance(v, Path): v = f"file://{v}"
            for x in (v if isinstance(v, list) else [v]):
                assert isinstance(x, PIL.Image.Image) or any(x.startswith(prefix) for prefix in ("http", "file://"))
        elif t == "text":
            assert isinstance(v, str)
        else:
            raise TypeError(f"Unsupported content type: {t}")
        ret.append({"type": t, t: v})
    return {"role": role, "content": ret}


def sv_annotate(image: np.ndarray,
                detections: sv.Detections,
                mask_opacity: float = 0.7,
                smart_label: bool = True) -> np.ndarray:
    """ :param image: OpenCV image
        :param detections: Supervision Detections with xyxy, confidence ..."""
    color_lookup = sv.ColorLookup.CLASS if detections.mask is None else sv.ColorLookup.INDEX
    anno_mask = sv.MaskAnnotator(color_lookup=color_lookup, opacity=mask_opacity)
    anno_box = sv.BoxAnnotator(color_lookup=color_lookup)
    anno_label = sv.LabelAnnotator(color_lookup=color_lookup, smart_position=smart_label)

    labels = [f"{score:.2f}" for score in detections.confidence] if detections.class_id is None else [
        f"{phrase} {score:.2f}" for phrase, score in zip(detections.metadata[detections.class_id], detections.confidence)]

    return anno_label.annotate(
        anno_box.annotate(
            anno_mask.annotate(image.copy(), detections=detections),
            detections=detections),
        detections=detections, labels=labels)
