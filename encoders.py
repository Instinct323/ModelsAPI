import torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_dinov2(size: str = "b",
                with_reg: bool = True):
    """ https://github.com/facebookresearch/dinov2
        :return DinoVisionTransformer
            forward(x) -> [B, C]
            get_intermediate_layers(x, n, reshape=True) -> n * [B, N, h, w]"""
    repo = "facebookresearch/dinov2"
    assert size in "sblg"
    return torch.hub.load(repo, f"dinov2_vit{size}14" + "_reg" * with_reg).eval().to(DEVICE)


class OpenCLIP:
    """ https://github.com/mlfoundations/open_clip
        :param pretrained: see open_clip/pretrained.py"""

    def __init__(self,
                 model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k"):
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval().to(DEVICE)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_images(self, *images):
        # [B, C, H, W] -> [B, C]
        x = torch.stack(list(map(self.preprocess, images))).to(DEVICE)
        x = self.model.encode_image(x)
        return x / x.norm(dim=-1, keepdim=True)

    def encode_texts(self, *texts):
        # [B, L] -> [B, C]
        x = self.tokenizer(texts).to(DEVICE)
        x = self.model.encode_text(x)
        return x / x.norm(dim=-1, keepdim=True)


if __name__ == '__main__':
    import PIL.Image

    clip = OpenCLIP()
    img = PIL.Image.open("assets/cat.jpg")
    img = clip.encode_images(img)
    text = clip.encode_texts("a cat", "a dog")
    print(img @ text.T)
