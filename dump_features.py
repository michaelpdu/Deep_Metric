#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Description: 
# Date: 2021/12/9 12:22
# Author: dupei

import argparse
import pickle
import torch
import DataSet
from evaluations import extract_features
from util import load_model


def dump_features(args):
    model = load_model(ckp_file=args.checkpoint, net=args.net, dim=args.dim)
    data = DataSet.create_general_dataset(root=args.data_root)

    gallery_loader = torch.utils.data.DataLoader(
        data.gallery, batch_size=args.batch_size, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=args.nThreads)
    feature, _ = extract_features(model, gallery_loader, print_freq=1e5, metric=None)

    with open(args.output, 'wb') as handle:
        pickle.dump(feature, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--data_root', type=str, default='data/Oxford')
    parser.add_argument('--output', type=str, default='oxford_gallery_feature.pickle')
    parser.add_argument('--net', type=str, default='BN-Inception')
    parser.add_argument('--checkpoint', type=str, default='ckp_ep600.pth.tar', metavar='PATH')
    parser.add_argument('--dim', '-d', type=int, default=512, help='Dimension of Embedding Feather')
    parser.add_argument('--width', type=int, default=227, help='width of input image')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--nThreads', '-j', default=8, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    return parser.parse_args()


if __name__ == '__main__':
    dump_features(parse_args())
