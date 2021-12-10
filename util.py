#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Description: 
# Date: 2021/12/9 10:25
# Author: dupei

import os
import shutil
import torch
import models
from DataSet.GeneralData import default_loader
from evaluations import pairwise_similarity
from utils.serialization import load_checkpoint
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


def load_image(image_path, width=227, origin_width=256):
    """
    @image_path, path to input image
    @return, tensor value, BGR image, shape is [C,H,W]
    """
    assert os.path.exists(image_path)
    img = default_loader(image_path)
    if img.height == img.width and img.width == width:
        transform_func = transforms.Compose([
            transforms.CovertBGR(),
            transforms.ToTensor()])
        return transform_func(img)
    else:
        transform_func = transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize(origin_width),
            transforms.CenterCrop(width),
            transforms.ToTensor()])
        return transform_func(img)


def loss_func(fea, target_fea):
    return pairwise_similarity(fea, target_fea)


def convert_bgr_to_rgb(pil_img):
    b, g, r = pil_img.split()
    return Image.merge("RGB", (r, g, b))


def save_image(tensor, save_path):
    to_pil_image = transforms.ToPILImage()
    pil_img = to_pil_image(tensor)
    pil_img = convert_bgr_to_rgb(pil_img)
    pil_img.save(save_path)


def create_model(net, dim, checkpoint):
    model = models.create(net, dim=dim, pretrained=False)
    # resume = load_checkpoint(ckp_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def load_model(ckp_file, net, dim):
    checkpoint = load_checkpoint(ckp_file)

    # 修改state_dict中的keys，去除module.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for name, param in checkpoint['state_dict'].items():
        new_state_dict[name.replace('module.', '')] = param
    checkpoint['state_dict'] = new_state_dict

    return create_model(net=net, dim=dim, checkpoint=checkpoint)


def extract_feature(model, inputs):
    """
    @inputs, tensor value, BGR image, shape is [C,H,W]
    @return, feature
    """
    model.eval()
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std=[1.0 / 255, 1.0 / 255, 1.0 / 255])
    inputs = normalize(inputs)
    inputs = torch.unsqueeze(inputs, dim=0)
    return model(inputs)


def pair_dist(model, img_file1, img_file2):
    f1 = extract_feature(model, load_image(img_file1))
    f2 = extract_feature(model, load_image(img_file2))
    dist = loss_func(f1, f2)
    print(dist)


def is_image_ext(file_path):
    image_ext_list = ['.jpg', '.png', '.jpeg', '.bmp']
    _, ext = os.path.splitext(file_path)
    return True if ext.lower() in image_ext_list else False