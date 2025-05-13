import warnings

warnings.filterwarnings("ignore")

import cv2
import requests

import groundingdino.datasets.transforms as T
import numpy as np
from PIL import Image

from groundingdino.util import inference

transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class GroundingDINO:

    def __init__(self,
                 box_thresh=0.35,
                 text_thresh=0.25):
        # Check connection
        assert requests.get("https://huggingface.co", timeout=5).status_code == 200
        # pip install git+https://github.com/IDEA-Research/GroundingDINO.git
        from groundingdino.config import GroundingDINO_SwinT_OGC as config
        checkpoint = "checkpoints/groundingdino_swint_ogc.pth"
        self.model = inference.load_model(config.__file__, checkpoint)
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh

    def __call__(self,
                 image: Image,
                 prompt: str):
        """ Open-vocabulary object detection
            :param image: PIL image
            :param prompt: text prompt (e.g. "computer. jar. cup")
            :return: boxes, logits, phrases """
        return inference.predict(
            model=self.model,
            image=transform(image.convert("RGB"), None)[0],
            caption=prompt,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh
        )


if __name__ == "__main__":
    gdino = GroundingDINO()

    # Inference
    image = Image.open("assets/color.png")
    ret = boxes, logits, phrases = gdino(image, "computer. jar. cup")
    print(ret)

    # Visualize
    annotated_frame = inference.annotate(image_source=np.asarray(image), boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("runs/test.jpg", annotated_frame)
