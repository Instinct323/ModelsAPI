# Leaderboard

- [Chatbot](https://openlm.ai/chatbot-arena/)
- [LLM Benchmark](https://livebench.ai/#/)
- [Zero-Shot Object Detection](https://paperswithcode.com/sota/zero-shot-object-detection-on-mscoco)

# Project Structure

- `depth_anything_v2`: from [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- `sam2`: from [facebookresearch/sam2](https://github.com/facebookresearch/sam2)

# Installation

Select preferences and run the command to install [PyTorch](https://pytorch.org/get-started/previous-versions/) locally

- CLIP

```bash
pip install open_clip_torch
```

- Depth Anything V2

```bash
pip install gradio_imageslider gradio==4.29.0
```

- Grounding DINO

```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

- Qwen-VL

```bash
pip install accelerate huggingface_hub[hf_xet] qwen-vl-utils[decord] transformers==4.50.3
```

- Segment Anything Model 2

```bash
# TODO
```

- Web

```bash
pip install fastapi requests uvicorn
```
