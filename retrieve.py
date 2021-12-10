#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Description: 
# Date: 2021/12/8 10:59
# Author: dupei

import argparse
import os
import pickle
import platform
import shutil
import torch
import DataSet
from evaluations import pairwise_similarity
from util import extract_feature, load_image, create_model
from utils.serialization import load_checkpoint


def save_retrieved_images(gallery, topk_index, topk_sim, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(topk_index)):
        image_path = gallery[topk_index[i]][2]
        _, filename = os.path.split(image_path)
        name, ext = os.path.splitext(filename)
        shutil.copyfile(image_path, os.path.join(output_dir, '%d_' % i + name + '_%f' % topk_sim[i] + ext))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Testing')

    parser.add_argument('--data', type=str, default='cub')
    parser.add_argument('--data_root', type=str, default='data/Oxford')

    parser.add_argument('--net', type=str, default='BN-Inception')
    parser.add_argument('--resume', '-r', type=str, default='ckp_ep600.pth.tar', metavar='PATH')

    parser.add_argument('--dim', '-d', type=int, default=512,
                        help='Dimension of Embedding Feather')
    parser.add_argument('--width', type=int, default=227,
                        help='width of input image')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--nThreads', '-j', default=8, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')

    parser.add_argument('--topk', '-k', type=int, default=10, help='top-k')
    parser.add_argument('--query_img', '-query', type=str, help='path to query image')
    parser.add_argument('--gallery_pickle', '-gallery', type=str, help='path to gallery image feature pickle file',
                        default='oxford_gallery_feature.pickle')
    parser.add_argument('--output', type=str, default='tmp/retrieved', help='path to output dir')

    args = parser.parse_args()
    return args


def main(args):
    if platform.system() == 'Windows':
        torch.multiprocessing.freeze_support()

    checkpoint = load_checkpoint(args.resume)

    # 修改state_dict中的keys，去除module.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for name, param in checkpoint['state_dict'].items():
        new_state_dict[name.replace('module.', '')] = param
    checkpoint['state_dict'] = new_state_dict

    model = create_model(net=args.net, dim=args.dim, checkpoint=checkpoint)

    # data = DataSet.create(args.data, width=args.width, root=args.data_root)
    data = DataSet.create_general_dataset(root=args.data_root)

    # gallery_loader = torch.utils.data.DataLoader(
    #     data.gallery, batch_size=args.batch_size, shuffle=False,
    #     drop_last=False, pin_memory=True, num_workers=args.nThreads)
    # gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None)

    with open(args.gallery_pickle, 'rb') as handle:
        gallery_feature = pickle.load(handle)

    # query_loader = torch.utils.data.DataLoader(
    #     data.query, batch_size=args.batch_size,
    #     shuffle=False, drop_last=False,
    #     pin_memory=True, num_workers=args.nThreads)
    # query_feature, query_labels = extract_features(model, query_loader, print_freq=1e5, metric=None)

    images = load_image(args.query_img)
    query_feature = extract_feature(model, images)

    sim_mat = pairwise_similarity(query_feature, gallery_feature)
    topk_result = torch.topk(sim_mat, args.topk, dim=1, largest=True, sorted=True)
    topk_sim = topk_result[0]
    topk_index = topk_result[1]

    topk_sim = torch.squeeze(topk_sim).detach().numpy()
    topk_index = torch.squeeze(topk_index).detach().numpy()

    os.makedirs(args.output, exist_ok=True)
    save_retrieved_images(data.gallery, topk_index, topk_sim, args.output)

# --data_root data\CUB_200_2011 --query_img data\CUB_200_2011\train\010.Red_winged_Blackbird\Red_Winged_Blackbird_0001_3695.jpg --gallery_pickle cub_train_feature.pickle --output tmp\cub_retrieved -k 20
# --data_root data/Oxford --query_img data/Oxford/query/all_souls_000013.jpg --output tmp/retrieved -k 20
if __name__ == '__main__':
    main(parse_args())
