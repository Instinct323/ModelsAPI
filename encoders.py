import torch


def load_dinov2(size: str = "b",
                with_reg: bool = True):
    # https://github.com/facebookresearch/dinov2
    repo = "facebookresearch/dinov2"
    assert size in "sblg"
    return torch.hub.load(repo, f"dinov2_vit{size}14" + "_reg" * with_reg)


if __name__ == '__main__':
   print(load_dinov2())