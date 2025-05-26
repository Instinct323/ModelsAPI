import torch
from PIL import Image

from utils import huggingface_model_path

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_dinov2(size: str = "b",
                with_reg: bool = True):
    """
    https://github.com/facebookresearch/dinov2
    :return DinoVisionTransformer
        forward(x) -> [B, C]
        get_intermediate_layers(x, n, reshape=True) -> n * [B, N, h, w]
    """
    repo = "facebookresearch/dinov2"
    assert size in "sblg"
    return torch.hub.load(repo, f"dinov2_vit{size}14" + "_reg" * with_reg).eval().to(DEVICE)


class OpenCLIP:
    """
    :param model_name: see open_clip/pretrained.py
    :param repo_id: see https://huggingface.co/laion?sort_models=likes#models
    """

    def __init__(self,
                 model_name: str = "ViT-B-32",
                 repo_id: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=str(next(huggingface_model_path(repo_id).iterdir())))
        self.model.eval().to(DEVICE)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_images(self, *images):
        """
        Encode images
        :param images: a list of PIL images
        :return: feature vectors with shape [B, C]
        """
        # [B, C, H, W] -> [B, C]
        x = torch.stack(list(map(self.preprocess, images))).to(DEVICE)
        x = self.model.encode_image(x)
        return x / x.norm(dim=-1, keepdim=True)

    def encode_texts(self, *texts):
        """
        Encode texts
        :param texts: a list of texts
        :return: feature vectors with shape [B, C]
        """
        # [B, L] -> [B, C]
        x = self.tokenizer(texts).to(DEVICE)
        x = self.model.encode_text(x)
        return x / x.norm(dim=-1, keepdim=True)


if __name__ == '__main__':
    # print(load_dinov2())

    clip = OpenCLIP()
    img = Image.open("assets/cat.jpg")

    with torch.no_grad():
        feat_img = clip.encode_images(img)
        feat_text = clip.encode_texts("a cat", "a dog")
    print(feat_img @ feat_text.T)
