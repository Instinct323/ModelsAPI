from functools import partial
from typing import Union, Tuple, List

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
        patches_range = patches_range or (None,) * 2
        self.processor: Qwen2_5_VLProcessor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path, use_fast=True,
            min_pixels=patches_range[0] * self.patch_size ** 2,
            max_pixels=patches_range[1] * self.patch_size ** 2
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, device_map=device_map
        )
        # 冻结模型参数
        for k, v in self.model.named_parameters(): v.requires_grad = False

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
                 requires_grad: bool = False,
                 simplify: bool = True):
        generate = self.model.generate
        if requires_grad: generate = partial(generate.__wrapped__, self.model)

        generated_ids = generate(**inputs, max_new_tokens=max_new_tokens)
        ids = [outi[len(ini):] for ini, outi in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(ids, skip_special_tokens=simplify, clean_up_tokenization_spaces=False)

    def chat(self,
             messages: list,
             max_new_tokens: int):
        messages = messages.copy()
        messages.append(
            make_content("assistant", self.generate(self.get_input_tensor(messages), max_new_tokens))[0]
        )
        return messages


if __name__ == '__main__':
    model = QwenVL("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16)
