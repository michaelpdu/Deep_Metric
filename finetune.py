# coding=utf-8
from __future__ import absolute_import, print_function
import time
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models
import losses
from util import extract_feature
from utils import FastRandomIdentitySampler, mkdir_if_missing, logging, display
from utils.serialization import save_checkpoint, load_checkpoint
from finetuner import train
from utils import orth_reg

import DataSet
import numpy as np
import os.path as osp

cudnn.benchmark = True

use_gpu = torch.cuda.is_available()


# Batch Norm Freezer : bring 2% improvement on CUB 
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def main(args):
    # s_ = time.time()
    save_dir = args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    sys.stdout = logging.Logger(os.path.join(save_dir, 'log.txt'))
    display(args)
    start = 0

    bn_inception_path = os.path.join(args.pretrained_model_dir, 'bn_inception-52deb4733.pth')
    assert os.path.exists(bn_inception_path), 'Cannot find pre-trained bn_inception model, %s' % bn_inception_path
    model = models.create(args.net, pretrained=True, dim=args.dim, model_path=bn_inception_path)

    # for vgg and densenet
    if args.resume is None:
        model_dict = model.state_dict()

    else:
        # resume model
        print('load model from {}'.format(args.resume))
        chk_pt = load_checkpoint(args.resume)
        # weight = chk_pt['state_dict']
        # start = chk_pt['epoch']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for name, param in chk_pt['state_dict'].items():
            new_state_dict[name.replace('module.', '')] = param
        model.load_state_dict(new_state_dict)

    model = torch.nn.DataParallel(model)
    if use_gpu:
        model = model.cuda()

    # freeze BN
    if args.freeze_BN is True:
        print(40 * '#', '\n BatchNorm frozen')
        model.apply(set_bn_eval)
    else:
        print(40 * '#', 'BatchNorm NOT frozen')

    # Fine-tune the model: the learning rate for pre-trained parameter is 1/10
    new_param_ids = set(map(id, model.module.classifier.parameters()))

    new_params = [p for p in model.module.parameters() if
                  id(p) in new_param_ids]

    base_params = [p for p in model.module.parameters() if
                   id(p) not in new_param_ids]

    param_groups = [
        {'params': base_params, 'lr_mult': 0.0},
        {'params': new_params, 'lr_mult': 1.0}]

    print('initial model is save at %s' % save_dir)

    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                 weight_decay=args.weight_decay)

    criterion = losses.create(args.loss, margin=args.margin, alpha=args.alpha, base=args.loss_base)
    if use_gpu:
        criterion = criterion.cuda()

    data = DataSet.create_target_triplet_dataset(root=args.data_root)

    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.batch_size,
        shuffle=True,
        # sampler=FastRandomIdentitySampler(data.train, num_instances=args.num_instances),
        drop_last=True, pin_memory=True, num_workers=args.nThreads)

    # save the train information
    for epoch in range(start, args.epochs):
        train(epoch=epoch, model=model, criterion=criterion,
              optimizer=optimizer, train_loader=train_loader, args=args)

    #     if epoch == 1:
    #         optimizer.param_groups[0]['lr_mul'] = 0.1
    #
    #     if (epoch + 1) % args.save_step == 0 or epoch == 0:
    #         if use_gpu:
    #             state_dict = model.module.state_dict()
    #         else:
    #             state_dict = model.state_dict()
    #
    #         save_checkpoint({
    #             'state_dict': state_dict,
    #             'epoch': (epoch + 1),
    #         }, is_best=False, fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Metric Learning')

    # hype-parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of new parameters")
    parser.add_argument('--batch_size', '-b', default=32, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('--num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('--dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('--width', default=224, type=int,
                        help='width of input image')
    parser.add_argument('--origin_width', default=256, type=int,
                        help='size of origin image')
    parser.add_argument('--ratio', default=0.16, type=float,
                        help='random crop ratio for train data')

    parser.add_argument('--alpha', default=30, type=int, metavar='n',
                        help='hyper parameter in NCA and its variants')
    parser.add_argument('--beta', default=0.1, type=float, metavar='n',
                        help='hyper parameter in some deep metric loss functions')
    parser.add_argument('--orth_reg', default=0, type=float,
                        help='hyper parameter coefficient for orth-reg loss')
    parser.add_argument('-k', default=16, type=int, metavar='n',
                        help='number of neighbour points in KNN')
    parser.add_argument('--margin', default=0.5, type=float,
                        help='margin in loss function')
    parser.add_argument('--init', default='random',
                        help='the initialization way of FC layer')

    # network
    parser.add_argument('--freeze_BN', default=True, type=bool, metavar='N',
                        help='Freeze BN if True')
    parser.add_argument('--data', default='myself',
                        help='name of Data Set')
    parser.add_argument('--data_root', type=str, default='data/DP',
                        help='path to Data Set')
    parser.add_argument('--pretrained_model_dir', type=str, default='bn_inception',
                        help='path to pre-trained bn_inception root dir')

    parser.add_argument('--net', default='BN-Inception')
    parser.add_argument('--loss', default='LiftedStructure',
                        help='loss for training network')
    parser.add_argument('--epochs', default=600, type=int, metavar='N',
                        help='epochs for training process')
    parser.add_argument('--save_step', default=50, type=int, metavar='N',
                        help='number of epochs to save model')

    # Resume from checkpoint
    parser.add_argument('--resume', '-r', default='ckp_ep600.pth.tar',
                        help='the path of the pre-trained model')

    # train
    parser.add_argument('--print_freq', default=20, type=int,
                        help='display frequency of training')

    # basic parameter
    # parser.add_argument('--checkpoints', default='/opt/intern/users/xunwang',
    #                     help='where the trained models save')
    parser.add_argument('--save_dir', default='checkpoints',
                        help='where the trained models save')
    parser.add_argument('--nThreads', '-j', default=8, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    parser.add_argument('--loss_base', type=float, default=0.75)

    main(parser.parse_args())
