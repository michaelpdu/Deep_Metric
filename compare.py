#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Description: 
# Date: 2021/12/9 10:40
# Author: dupei
import argparse
import os

from util import load_model, pair_dist


class CompareHelper:
    def __init__(self, checkpoint, net, dim):
        self.model = load_model(checkpoint, net, dim)

    def compare(self, query_img, gallery_img):
        print('gallery_img:', gallery_img)
        pair_dist(self.model, query_img, gallery_img)

    def compare_dir(self, query_img, gallery_dir):
        for root, dirs, files in os.walk(gallery_dir):
            for name in files:
                gallery_img = os.path.join(root, name)
                self.compare(query_img, gallery_img)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--query_img', type=str)
    parser.add_argument('--gallery_img', type=str, default=None)
    parser.add_argument('--gallery_dir', type=str, default=None)
    parser.add_argument('--net', type=str, default='BN-Inception')
    parser.add_argument('--checkpoint', '-r', type=str, default='ckp_ep600.pth.tar', metavar='PATH')
    parser.add_argument('--dim', '-d', type=int, default=512, help='Dimension of Embedding Feather')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.query_img is not None and os.path.exists(args.query_img)

    helper = CompareHelper(args.checkpoint, args.net, args.dim)
    if args.gallery_img is not None and os.path.exists(args.gallery_img):
        helper.compare(args.query_img, args.gallery_img)
    elif args.gallery_dir is not None and os.path.exists(args.gallery_dir) and os.path.isdir(args.gallery_dir):
        helper.compare_dir(args.query_img, args.gallery_dir)
    else:
        print('cannot find  gallery image or dir')
