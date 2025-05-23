import os
from functools import partial

import cv2
import openai

from utils import ContentMaker


class AIclient:

    def __init__(self,
                 model: str,
                 api_key: str,
                 base_url: str):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.cont_maker = ContentMaker(False)

    def query_once(self,
                   max_new_tokens: int,
                   **contents):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[self.cont_maker("user", **contents)],
            max_tokens=max_new_tokens
        ).choices[0].message.content

    def __repr__(self):
        return self.model


# Qwen: https://help.aliyun.com/model-studio/getting-started/models
QwenClient = partial(AIclient,
                     api_key=os.getenv("DASHSCOPE_API_KEY"),
                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

if __name__ == '__main__':
    qwen = QwenClient("qwen-vl-max-latest")

    print(qwen.query_once(128, text="这些图描绘了什么内容？", image=cv2.imread("assets/color.png")))
