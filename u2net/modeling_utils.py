import os
import requests
from typing import *

import torch
import torchvision
from PIL import Image

from .models import U2NET


def download_model(model_path: str, ):
    url = "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ&export=download&confirm=t"

    session = requests.session()
    response = session.get(
        url,
        allow_redirects=True,
    )

    assert response.ok, "Error downloading pre-trained weights!"

    with open(model_path, "wb") as ckpt_file:
        ckpt_file.write(response.content)


def normalize_prediction(pred: torch.Tensor, ) -> torch.Tensor:
    max_pred = torch.max(pred)
    min_pred = torch.min(pred)

    norm_pred = (pred - min_pred) / (max_pred - min_pred)

    return norm_pred


def get_img_mask(
    img: Image.Image,
    model_name: str = "u2net",
    device: str = "cuda:0",
) -> Union[torch.Tensor, Image.Image]:
    model_path = os.path.join(
        os.getcwd(),
        'saved_models',
        model_name,
        model_name + '.pth',
    )

    if not os.path.exists(model_path):
        download_model(model_path)

    if torch.is_tensor(img):
        img_tensor = img

    else:
        img_tensor = torchvision.transforms.PILToTensor()(img)
        img_tensor = (img_tensor / 255.) * 2 - 1
        img_tensor = img_tensor.to(device, torch.float32)[None, :]

    u2net = U2NET(3, 1)
    if torch.cuda.is_available():
        u2net.load_state_dict(torch.load(model_path))
        u2net.to(device, )
    else:
        u2net.load_state_dict(torch.load(model_path, map_location='cpu'))

    u2net.eval()

    d1, d2, d3, d4, d5, d6, d7 = u2net(img_tensor, )

    # NOTE: useful code to visualize all masks
    # for idx, d in enumerate([d1, d2, d3, d4, d5, d6, d7]):
    #     pred = d[:, 0, :, :]
    #     pred = normalize_prediction(pred, )

    #     mask = torchvision.transforms.ToPILImage()(pred)
    #     mask.save(f"{idx}.png")

    pred = d2[:, 0, :, :]
    pred = normalize_prediction(pred, )

    if torch.is_tensor(img):
        mask = pred[None, :]

    else:
        mask = torchvision.transforms.ToPILImage()(pred)

    return mask


if __name__ == "__main__":
    # img = Image.open("fungi.jpg").convert("RGB", )
    img = Image.open("ape.png").convert("RGB", )
    img = img.resize((128, 128))
    mask = get_img_mask(img=img, )
    mask.save("mask.png")
