#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Description: 
# Date: 2021/12/9 12:22
# Author: dupei

import pickle


def load_features():
    with open('gallery_feature.pickle', 'rb') as handle:
        feature = pickle.load(handle)
    print(feature)


if __name__ == '__main__':
    load_features()
