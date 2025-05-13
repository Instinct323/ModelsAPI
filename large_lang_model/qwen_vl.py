import time
from typing import Union, Tuple, List

import requests
import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLProcessor, Qwen2VLImageProcessorFast

from utils import *


class QwenVL:
    patch_size = Qwen2VLImageProcessorFast.patch_size
    device = property(lambda self: self.model.device)

    def __init__(self,
                 pretrained_model_name_or_path: str,
                 patches_range: Tuple[int, int] = (16, 512),
                 torch_dtype: torch.dtype = "auto",
                 device_map: Union[str, torch.device] = "auto"):
        import transformers
        version = list(map(int, transformers.__version__.split(".")))
        assert version[0] == 4 and version[1] <= 50, "[2025-05-12] transformers version must be 4.50.x or lower"
        # Check connection
        assert requests.get("https://huggingface.co", timeout=5).status_code == 200

        t0 = time.time()
        patches_range = patches_range or (None,) * 2
        self.processor: Qwen2_5_VLProcessor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path, use_fast=True,
            min_pixels=patches_range[0] * self.patch_size ** 2,
            max_pixels=patches_range[1] * self.patch_size ** 2
        )
        LOGGER.info(f"Processor loaded in {time.time() - t0:.2f}s.")

        t0 = time.time()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, device_map=device_map
        )
        LOGGER.info(f"Model loaded in {time.time() - t0:.2f}s.")

    def get_input_tensor(self,
                         messages,
                         batch_inference: bool = False):
        # fixme: batch_inference 不能正常使用
        texts: List[str] = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                            for msg in (messages if batch_inference else [messages])]
        # List[PIL.Image.Image]
        images, videos = process_vision_info(messages)
        # transformers.feature_extraction_utils.BatchFeature
        #   `input_ids`: 输入文本的 token ID
        #   `attention_mask`: bool mask, 用于指示哪些 token 是非填充的
        #   `pixel_values`: 输入图像的像素值
        #   `image_grid_thw`: 时间维度, 高度、宽度上的 patch 数量
        return self.processor(text=texts, images=images, videos=videos, padding=True, return_tensors="pt").to(self.device)

    def generate(self,
                 inputs,
                 max_new_tokens: int,
                 simplify: bool = True):
        """ Generate text from the model.
            :param inputs: The input tensor from get_input_tensor.
            :param max_new_tokens: The maximum number of new tokens to generate.
            :return: The generated text. """
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        ids = [outi[len(ini):] for ini, outi in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(ids, skip_special_tokens=simplify, clean_up_tokenization_spaces=False)

    def query_once(self,
                   max_new_tokens: int,
                   **contents):
        """ Query the model once.
            :param max_new_tokens: The maximum number of new tokens to generate.
            :param contents: The input contents (e.g., text, image, video)."""
        return self.generate(self.get_input_tensor([
            make_content("user", **contents)
        ]), max_new_tokens=max_new_tokens)[0]

    def chat(self,
             max_new_tokens: int):
        """ Chat with the model. Type 'exit' to quit."""
        messages = []
        while True:
            msg = input("User: ")
            if msg == "exit": break
            messages.append(make_content("user", text=msg))
            messages.append(make_content("assistant", text=self.generate(self.get_input_tensor(messages), max_new_tokens)[0]))

            text = messages[-1]["content"]
            text = text[0] if isinstance(text, list) else text
            text = text["text"] if isinstance(text, dict) else text
            yield text


if __name__ == '__main__':
    model = QwenVL("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16)

    print(model.query_once(512, text="描述这张图片",
                           image="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"))

    for ctx in model.chat(512): print(ctx)
