from __future__ import absolute_import, print_function


"""
Triplet data-set for Pytorch
"""

import os
from PIL import Image
from DataSet import transforms
import torch.utils.data as data


def default_loader(path):
    return Image.open(path).convert('RGB')


def generate_transform_dict(origin_width=256, width=227, ratio=0.16):
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std=[1.0 / 255, 1.0 / 255, 1.0 / 255])

    transform_dict = {'rand-crop': transforms.Compose([
        transforms.CovertBGR(),
        transforms.Resize(origin_width),
        transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), 'center-crop': transforms.Compose([
        transforms.CovertBGR(),
        transforms.Resize(origin_width),
        transforms.CenterCrop(width),
        transforms.ToTensor(),
        normalize,
    ]), 'resize': transforms.Compose([
        transforms.CovertBGR(),
        transforms.Resize(width),
        transforms.ToTensor(),
        normalize,
    ])}

    return transform_dict


class MyData(data.Dataset):
    def __init__(self, root, triplet_txt, transform, loader=default_loader):
        assert os.path.exists(root)
        assert os.path.exists(triplet_txt)
        assert transform is not None

        self.root = root
        self.transform = transform
        self.loader = loader
        self.image_triplet_list = []  # element is (query_img, retrieved1, retrieved2)

        with open(triplet_txt, 'r') as fh:
            for line in fh.readlines():
                path_list = line.strip().split(',')
                self.build_triplet(path_list)

        self.cache = {}

    def build_triplet(self, path_list):
        base_img_path = path_list[0]
        size = len(path_list)
        for i in range(1, size):
            for j in range(i+1, size):
                self.image_triplet_list.append((base_img_path, path_list[i], path_list[j]))

    def __getitem__(self, index):
        image_triplet = self.image_triplet_list[index]
        assert len(image_triplet) == 3

        if image_triplet[0] in self.cache.keys():
            img = self.cache[image_triplet[0]]
        else:
            img = self.transform(self.loader(os.path.join(self.root, image_triplet[0])))
            self.cache[image_triplet[0]] = img

        if image_triplet[1] in self.cache:
            img_pos = self.cache[image_triplet[1]]
        else:
            img_pos = self.transform(self.loader(os.path.join(self.root, image_triplet[1])))
            self.cache[image_triplet[1]] = img_pos

        if image_triplet[2] in self.cache:
            img_neg = self.cache[image_triplet[2]]
        else:
            img_neg = self.transform(self.loader(os.path.join(self.root, image_triplet[2])))
            self.cache[image_triplet[2]] = img_neg

        return img, img_pos, img_neg

    def __len__(self):
        return len(self.image_triplet_list)


class TripletData:
    def __init__(self, root, width=227, origin_width=256, ratio=0.16):
        transform_dict = generate_transform_dict(origin_width=origin_width, width=width, ratio=ratio)
        train_txt = os.path.join(root, 'train.txt')
        self.train = MyData(root, triplet_txt=train_txt, transform=transform_dict['center-crop'])
        # test_txt = os.path.join(root, 'test.txt')
        # self.test = MyData(root, label_txt=test_txt, transform=transform_dict['center-crop'])


def test_triplet_data():
    print(TripletData.__name__)
    data = TripletData("data/DP/")
    print(len(data.train))
    print(data.train[1])


if __name__ == "__main__":
    test_triplet_data()
