#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Description: 
# Date: 2021/12/6 9:21
# Author: dupei


import os
import torch

DATA = 'cub'
DATA_ROOT = 'data'
Gallery_eq_Query = True
LOSS = 'LiftedStructure'
CHECKPOINTS = 'ckps'
R = '.pth.tar'

os.makedirs(os.path.join(CHECKPOINTS, LOSS, DATA), exist_ok=True)
os.makedirs(os.path.join('result', LOSS, DATA), exist_ok=True)

NET = 'BN-Inception'
DIM = 512
ALPHA = 40
LR = 1e-5
BatchSize = 80
RATIO = 0.16

SAVE_DIR = os.path.join(CHECKPOINTS, LOSS, DATA, '%s-dim%d-lr%f-ratio%f-batch%d' % (NET, DIM, LR, RATIO, BatchSize))
os.makedirs(SAVE_DIR, exist_ok=True)

print("Begin Training!")

cmd = ''
if torch.cuda.is_available():
    cmd = cmd + 'CUDA_VISIBLE_DEVICES=0 '

cmd = cmd + 'python train.py' \
      + ' --net {}'.format(NET) \
      + ' --data {}'.format(DATA) \
      + ' --data_dir {}'.format(DATA_ROOT) \
      + ' --lr {}'.format(LR) \
      + ' --dim {}'.format(DIM) \
      + ' --alpha {}'.format(ALPHA) \
      + ' --num_instances {}'.format(5) \
      + ' --batch_size {}'.format(BatchSize) \
      + ' --epoch {}'.format(600) \
      + ' --loss {}'.format(LOSS) \
      + ' --width {}'.format(227) \
      + ' --save_dir {}'.format(SAVE_DIR) \
      + ' --save_step {}'.format(50) \
      + ' --ratio {}'.format(RATIO)
os.system(cmd)

print("End Training!")
