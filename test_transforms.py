#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Description: 
# Date: 2021/12/8 10:59
# Author: dupei


import os
import shutil

import torch
from DataSet.GeneralData import generate_transform_dict, default_loader
from evaluations import pairwise_similarity
from DataSet import transforms
from PIL import Image


def save_retrieved_images(gallery, topk_index, topk_sim, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(topk_index)):
        image_path = gallery[topk_index[i]][2]
        _, filename = os.path.split(image_path)
        name, ext = os.path.splitext(filename)
        shutil.copyfile(image_path, os.path.join(output_dir, '%d_' % i + name + '_%f' % topk_sim[i] + ext))


def load_image(image_path, width=227, origin_width=256, ratio=0.16):
    """"""
    assert os.path.exists(image_path)
    transform_dict = generate_transform_dict(origin_width=origin_width, width=width, ratio=ratio)
    img = default_loader(image_path)
    transform = transform_dict['center-crop']
    img = transform(img)
    return torch.unsqueeze(img, dim=0)


def convert_bgr_to_rgb(pil_img):
    b, g, r = pil_img.split()
    return Image.merge("RGB", (r, g, b))


def save_image(tensor, save_path):
    to_pil_image = transforms.ToPILImage()
    pil_img = to_pil_image(tensor)
    pil_img = convert_bgr_to_rgb(pil_img)
    pil_img.save(save_path)


def loss_func(fea, target_fea):
    return pairwise_similarity(fea, target_fea)


def main():
    img_path = 'tmp/retrieved/0_all_souls_000013_1.000000.jpg'
    img = default_loader(img_path)
    transform_func = transforms.Compose([
        transforms.CovertBGR(),
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor()])
    tensor = transform_func(img)

    _, filename = os.path.split(img_path)
    name, ext = os.path.splitext(filename)

    save_image(tensor, os.path.join('tmp', name + '_cropped.png'))

    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std=[1.0 / 255, 1.0 / 255, 1.0 / 255])
    tensor = normalize(tensor)
    save_image(tensor, os.path.join('tmp', name + '_cropped_normalized.png'))


if __name__ == '__main__':
    main()
