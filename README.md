# Leaderboard

- [Chatbot](https://openlm.ai/chatbot-arena/)
- [LLM Benchmark](https://livebench.ai/#/)
- [Zero-Shot Object Detection](https://paperswithcode.com/sota/zero-shot-object-detection-on-mscoco)

# Installation

Install the basic requirements:

```bash
pip install -r requirements.txt
```

Select preferences and run the command to install [PyTorch](https://pytorch.org/get-started/previous-versions/) locally.

- [CLIP](https://github.com/mlfoundations/open_clip)

```bash
pip install open_clip_torch
```

- [Contact-GraspNet](https://github.com/elchun/contact_graspnet_pytorch)

```bash
pip install provider pyrender
```

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)

```bash
pip install gradio_imageslider gradio==4.29.0
```

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO.git)

```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

- [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)

```bash
pip install accelerate huggingface_hub[hf_xet] qwen-vl-utils[decord] transformers==4.50.3
```

- [Segment Anything V2](https://github.com/facebookresearch/sam2)

```bash
pip install hydra-core iopath
```

- [VGGT](https://github.com/facebookresearch/vggt)

```bash
pip install git+https://github.com/facebookresearch/vggt
```

# Web

Install the following packages to run the web server:

```bash
pip install fastapi requests uvicorn
```

You need to create a Python file that stores the API in a dictionary format within a variable named `FUNCTIONS`, and set the file path in `server.py`.

Then, run the server using the command below:

```bash
uvicorn server:app
```

# Dataset

- [Graspnet API](https://github.com/graspnet/graspnetAPI.git)

```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
sed -i 's/sklearn/scikit-learn/g' setup.py
pip install .
```