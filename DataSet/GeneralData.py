from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
from PIL import Image

import os
import sys
from DataSet import transforms 
from collections import defaultdict


def default_loader(path):
    return Image.open(path).convert('RGB')


def generate_transform_dict(origin_width=256, width=227, ratio=0.16):
    
    std_value = 1.0 / 255.0
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std= [1.0/255, 1.0/255, 1.0/255])

    transform_dict = {}

    transform_dict['rand-crop'] = \
    transforms.Compose([
                transforms.CovertBGR(),
                transforms.Resize((origin_width)),
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
               ])

    transform_dict['center-crop'] = \
    transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.Resize((origin_width)),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])
    
    transform_dict['resize'] = \
    transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.Resize((width)),
                    transforms.ToTensor(),
                    normalize,
                ])
    return transform_dict


class MyData(data.Dataset):
    def __init__(self, root=None, label_txt=None,
                 transform=None, loader=default_loader):

        # Initialization data path and train(gallery or query) txt path
        if root is None:
            self.root = "data/cub/"
        self.root = root
        
        if label_txt is None:
            label_txt = os.path.join(root, 'train.txt')

        if transform is None:
            transform_dict = generate_transform_dict()['rand-crop']

        file = open(label_txt)
        images_anon = file.readlines()

        images = []
        labels = []

        for img_anon in images_anon:

            [img, label] = img_anon.split(' ')
            images.append(img)
            labels.append(int(label))

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.images = images
        self.labels = labels
        self.classes = classes
        self.transform = transform
        self.Index = Index
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.images[index], self.labels[index]
        fn = os.path.join(self.root, fn)
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, fn

    def __len__(self):
        return len(self.images)


class GeneralData:
    def __init__(self, width=227, origin_width=256, ratio=0.16, root=None, transform=None):
        print('width: \t {}'.format(width))
        transform_dict = generate_transform_dict(origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = "data/CUB_200_2011/"

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')
        gallery_txt = os.path.join(root, 'gallery.txt')
        query_txt = os.path.join(root, 'query.txt')

        self.train = MyData(root, label_txt=train_txt, transform=transform_dict['center-crop'])
        self.test = MyData(root, label_txt=test_txt, transform=transform_dict['center-crop'])
        self.gallery = MyData(root, label_txt=gallery_txt, transform=transform_dict['center-crop'])
        self.query = MyData(root, label_txt=query_txt, transform=transform_dict['center-crop'])


def testCUB_200_2011():
    print(GeneralData.__name__)
    data = GeneralData()
    print(len(data.gallery))
    print(len(data.train))
    print(data.train[1])


if __name__ == "__main__":
    testCUB_200_2011()


