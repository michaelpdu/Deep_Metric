from .CUB200 import CUB_200_2011
from .Car196 import Cars196
from .Products import Products
from .In_shop_clothes import InShopClothes
# from .transforms import *
import os

from .GeneralData import GeneralData
from .TripletData import TripletData

__factory = {
    'cub': CUB_200_2011,
    'car': Cars196,
    'product': Products,
    'shop': InShopClothes,
}


def names():
    return sorted(__factory.keys())


def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__


def create(name, root=None, *args, **kwargs):
    """
    Create a dataset instance.
    """
    if root is not None:
        root = os.path.join(root, get_full_name(name))

    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root=root, *args, **kwargs)


def create_general_dataset(root):
    return GeneralData(root=root)


def create_target_triplet_dataset(root):
    return TripletData(root=root)