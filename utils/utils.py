import logging
from pathlib import Path

import PIL.Image
import supervision as sv

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("utils")


def huggingface_model_path(repo_id: str) -> str:
    """ Download the model from Hugging Face. """
    path = Path(f"~/.cache/huggingface/hub/models--{repo_id.replace('/', '--')}/snapshots").expanduser()
    if not path.exists():
        LOGGER.warning(f"If the loading time is too long, run the following command to download the model: huggingface-cli download {repo_id}")
        return repo_id
    return str(next(path.iterdir()))


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
