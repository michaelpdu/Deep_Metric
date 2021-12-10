#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Description: 
# Date: 2021/12/8 10:59
# Author: dupei


import argparse
import ast
import os
import torch

from util import load_image, extract_feature, loss_func, save_image, load_model, pair_dist


class Attacker:
    def __init__(self, model, eps=0.3, alpha=2 / 255, steps=40, random_start=False):
        self.target_feature = None
        self.model = model
        self.use_gpu = torch.cuda.is_available()
        self.is_targeted = False
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def set_target_feature(self, target_feature):
        self.target_feature = target_feature
        self.is_targeted = True

    def set_target(self, target_img_file):
        images = load_image(target_img_file)
        fea0 = extract_feature(self.model, images)
        self.set_target_feature(fea0)

    def attack(self, img_file, adv_dir):
        os.makedirs(adv_dir, exist_ok=True)

        _, filename = os.path.split(img_file)
        name, _ = os.path.splitext(filename)

        images = load_image(img_file)
        save_image(images, os.path.join(adv_dir, '{}_cropped.png'.format(name)))

        fea0 = None
        if self.is_targeted:
            fea0 = self.target_feature
        else:
            fea0 = extract_feature(self.model, images)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True

            # Calculate loss
            fea = extract_feature(self.model, adv_images)
            loss = loss_func(fea, fea0)
            loss = -loss # loss是相似度，untargeted attack是为了让相似度尽可能减小
            if self.is_targeted:
                loss = -loss

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # Calculate loss for adv images
            fea = extract_feature(self.model, adv_images)
            loss = loss_func(fea, fea0)
            sim = loss.detach().numpy()[0][0]
            print('iteration:', i, ', loss:', sim)
            if loss < 0:
                print('loss is less than 0, and stop')
                break
            adv_img_path = os.path.join(adv_dir, '{}_adv_{}_{:.4f}.png'.format(name, i+1, sim))
            save_image(adv_images, adv_img_path)

            # new_adv_imgs = load_image(adv_img_path)
            # new_fea = extract_feature(self.model, new_adv_imgs)
            # loss_new = loss_func(fea, new_fea)
            # print('loss_new:', loss_new)

        return adv_images


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Testing')

    parser.add_argument('--data', type=str, default='cub')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--gallery_eq_query', '-g_eq_q', type=ast.literal_eval, default=False,
                        help='Is gallery identical with query')
    parser.add_argument('--net', type=str, default='BN-Inception')
    parser.add_argument('--resume', '-r', type=str, default='ckp_ep600.pth.tar', metavar='PATH')

    parser.add_argument('--dim', '-d', type=int, default=512,
                        help='Dimension of Embedding Feather')
    parser.add_argument('--width', type=int, default=227,
                        help='width of input image')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--nThreads', '-j', default=8, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--pool_feature', type=ast.literal_eval, default=False, required=False,
                        help='if True extract feature from the last pool layer')
    parser.add_argument('--topk', '-k', type=int, default=10, help='top-k')

    parser.add_argument('--input_img', '-input', type=str)
    parser.add_argument('--target_img', '-target', type=str, default=None)
    parser.add_argument('--adv_dir', type=str, default='tmp/adv_dir')

    args = parser.parse_args()
    return args


def test_pair_dist(args):
    model = load_model(args.resume, args.net, args.dim)
    for name in os.listdir('tmp/retrieved'):
        img_path = os.path.join('tmp/retrieved', name)
        pair_dist(model, 'tmp/retrieved/0_all_souls_000013_1.000000.jpg', img_path)


def test_pair_dist_adv(args):
    model = load_model(args.resume, args.net, args.dim)
    base_img_path = 'tmp/retrieved/0_all_souls_000013_1.000000.jpg'
    adv_img_path = 'tmp/adv_imgs/0_all_souls_000013_1.000000_adv_4.png'
    pair_dist(model, base_img_path, adv_img_path)


def test_grad(args):
    model = load_model(args.resume, args.net, args.dim)
    images = load_image('tmp/retrieved/0_all_souls_000013_1.000000.jpg')
    adv_images = images.clone().detach()
    adv_images.requires_grad = True
    fea = extract_feature(model, adv_images)
    loss = torch.mean(fea)
    loss.backward()
    print(adv_images.grad)


# --input_img data/Oxford/query/all_souls_000013.jpg --adv_dir tmp/adv_untargeted
# --input_img data/Oxford/query/all_souls_000013.jpg --target_img data/Oxford/gallery/worcester_000198.jpg --adv_dir tmp/adv_targeted

# --input_img data\CUB_200_2011\train\010.Red_winged_Blackbird\Red_Winged_Blackbird_0001_3695.jpg --adv_dir tmp/adv_cub_untargeted
# --input_img data\CUB_200_2011\train\010.Red_winged_Blackbird\Red_Winged_Blackbird_0001_3695.jpg --target_img data\CUB_200_2011\train\100.Brown_Pelican\Brown_Pelican_0124_93684.jpg --adv_dir tmp/adv_cub_targeted
def main(args):
    assert args.input_img is not None and os.path.exists(args.input_img)
    model = load_model(args.resume, args.net, args.dim)
    attacker = Attacker(model)
    if args.target_img is not None and os.path.exists(args.target_img):
        attacker.set_target(args.target_img)
    attacker.attack(args.input_img, args.adv_dir)


if __name__ == '__main__':
    # test_pair_dist(parse_args())
    # test_pair_dist_adv(parse_args())
    # test_grad(parse_args())
    main(parse_args())
