import os

import torch
import torchvision
from PIL import Image

from model import U2NET


def normalize_prediction(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def get_img_mask(
    img: Image.Image,
    model_name: str = "u2net",
    device: str = "cuda:0",
):
    model_dir = os.path.join(
        os.getcwd(),
        'saved_models',
        model_name,
        model_name + '.pth',
    )

    img_tensor = torchvision.transforms.PILToTensor()(img)
    img_tensor = (img_tensor / 255.) * 2 - 1
    img_tensor = img_tensor.to(device, torch.float32)[None, :]

    u2net = U2NET(3, 1)
    if torch.cuda.is_available():
        u2net.load_state_dict(torch.load(model_dir))
        u2net.to(device, )
    else:
        u2net.load_state_dict(torch.load(model_dir, map_location='cpu'))

    u2net.eval()

    d1, d2, d3, d4, d5, d6, d7 = u2net(img_tensor, )

    for idx, d in enumerate([d1, d2, d3, d4, d5, d6, d7]):
        pred = d[:, 0, :, :]
        pred = normalize_prediction(pred, )

        mask = torchvision.transforms.ToPILImage()(pred)
        mask.save(f"{idx}.png")

    return mask


if __name__ == "__main__":
    img = Image.open("fungi.jpg").convert("RGB", )
    mask = get_img_mask(img=img, )
