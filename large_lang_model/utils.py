import logging
from pathlib import Path

import PIL.Image

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def make_content(role, **contents) -> dict:
    """ :param role: user, assistant, system"""
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
