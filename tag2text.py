import torch
from PIL import Image
from ram import get_transform, inference_tag2text
from ram.models import tag2text


class Tag2Text:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self,
                 image_size: int = 384,
                 tag_thresh: float = 0.68):
        self.transform = get_transform(image_size=image_size)
        self.model = tag2text(pretrained="checkpoints/tag2text_swin_14m.pth",
                              image_size=image_size,
                              vit='swin_b',
                              threshold=tag_thresh).eval().to(self.device)

    def __call__(self,
                 image: Image):
        """ :returns: tags_en, tags_cn """
        image = self.transform(image).unsqueeze(0).to(self.device)
        res = inference_tag2text(image, self.model)
        return res[0], res[2]


if __name__ == '__main__':
    tt = Tag2Text()
    image = Image.open("assets/color.png")
    print(tt(image))
