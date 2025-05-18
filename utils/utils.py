import logging
from pathlib import Path

import PIL.Image
import supervision as sv

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("utils")


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


def detection_labels(detections: sv.Detections):
    if detections.class_id is not None:
        return [
            f"{phrase} {score:.2f}"
            for phrase, score
            in zip(detections.metadata[detections.class_id], detections.confidence)
        ]
    else:
        return [f"{score:.2f}" for score in detections.confidence]
