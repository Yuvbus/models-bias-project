from __future__ import division
from torchvision import models, transforms
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from modelvshuman import constants as c
from os.path import join as pjoin


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert("RGB")
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image.to(device)


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = set(['0', '5', '10', '19', '28'])
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


def content_loss(x, y):
    return F.mse_loss(x, y)


def gram_matrix(x):
    return torch.einsum('bchw,bdhw->bcd', x, x)


def style_loss(x, y):
    gx = gram_matrix(x)
    gy = gram_matrix(y)
    return F.mse_loss(gx, gy)


def total_variance_loss(x):
    a = torch.abs((x[:, :, 1:, :] - x[:, :, :-1, :])).sum()
    b = torch.abs((x[:, :, :, 1:] - x[:, :, :, :-1])).sum()
    return torch.pow(a + b, 1.125)


def create_stylized_image(
        content_path,
        style_path,
        new_image_path,
        total_step=1000,
        sample_step=100,
        content_weight=1,
        style_weight=10,
        total_variance_weight=0.01,
        lr=1.0
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    content = load_image(content_path, transform)
    style = load_image(style_path, transform)
    target = content.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([target], lr=lr)
    vgg = VGGNet().to(device).eval()

    content_features = vgg(content)
    style_features = vgg(style)

    for step in range(total_step):
        def closure():
            target_features = vgg(target)
            s_loss = 0
            c_loss = 0
            tv_loss = total_variance_loss(target) / (3 * content.size(2) * content.size(3))
            for f1, f2, f3 in zip(target_features, content_features, style_features):
                _, c, h, w = f1.size()
                c_loss += content_loss(f1, f2)
                s_loss += style_loss(f1, f3) / (c * h * w)
                tv_loss += total_variance_loss(f1) / (c * h * w)

            loss = (content_weight * c_loss + style_weight * s_loss + total_variance_weight * tv_loss)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            print(
                f'\rStep [{step + 1}/{total_step}], Content Loss: {c_loss.item():7.4f}, '
                f'Style Loss: {s_loss.item():10.4f}, Total Variance Loss: {tv_loss.item():10.4f}',
                end=''
            )
            return loss

        optimizer.step(closure)

        # if step == total_step - 1:
        if (step + 1) % sample_step == 0:
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.cpu().detach().squeeze()
            img = denorm(img).clamp_(0, 1)
            torchvision.utils.save_image(img, f"{new_image_path}_{step}.png")


if __name__ == "__main__":
    cue_conflict_folder = pjoin(c.DATASET_DIR, "cue-conflict1")
    Path(cue_conflict_folder).mkdir(parents=True, exist_ok=True)
    texture_folder_path = pjoin(c.DATASET_DIR, "texture")
    categories_images_dir = pjoin(c.DATASET_DIR, "white-background-filtered")
    for category_dir in os.listdir(categories_images_dir):
        if category_dir in ["airplane", "boat", "car", "keyboard", "oven", "truck"]:
            continue
        subdir_path = os.path.join(categories_images_dir, category_dir)
        cue_conflict_category = os.path.join(cue_conflict_folder, category_dir)
        Path(cue_conflict_category).mkdir(parents=True, exist_ok=True)
        if os.path.isdir(subdir_path):
            for i, image in enumerate(os.listdir(subdir_path)):
                image_path = os.path.join(subdir_path, image)
                for texture_path in list(Path(texture_folder_path).glob("**/*.*")):
                    texture = texture_path.stem
                    if texture.startswith(category_dir):
                        continue
                    new_image_name = f"{category_dir}{i}_{texture}"
                    new_image_path = os.path.join(cue_conflict_category, f"{new_image_name}")
                    create_stylized_image(image_path, texture_path, new_image_path)
                    print(new_image_name)
