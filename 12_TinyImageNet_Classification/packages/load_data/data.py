from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split

from .datasets import TINDataset, _Getdata

from .augmentation import MNIST_Transforms, CIFAR10_Transforms, CIFAR10_AlbumTrans, TinyImageNet_AlbumTrans

class MNISTDataLoader:
    """
    It creates a data loader for test and train. It taken transformations from the 'augmentation' module
    """
    def __init__(self, model_transform, batch_size=64, data_dir= './root', shuffle=True, nworkers=4, pin_memory=True):
        self.data_dir = data_dir

        self.train_set = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=model_transform.build_transforms(train=True)
        )

        self.test_set = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=model_transform.build_transforms(train=False)
        )

        self.init_kwargs = {
            'shuffle': shuffle,
            'batch_size': batch_size,
            'num_workers': nworkers,
            'pin_memory': pin_memory
        }


    def get_loaders(self):
        return DataLoader(self.train_set, **self.init_kwargs), DataLoader(self.test_set, **self.init_kwargs)

class CIFAR10DataLoader:

    class_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, model_transform, batch_size=64, data_dir= './root', shuffle=True, nworkers=4, pin_memory=True):
        self.data_dir = data_dir

        self.train_set = datasets.CIFAR10(
            self.data_dir,
            train=True,
            download=True,
            transform=model_transform.build_transforms(train=True)
        )

        self.test_set = datasets.CIFAR10(
            self.data_dir,
            train=False,
            download=True,
            transform=model_transform.build_transforms(train=False)
        )

        self.init_kwargs = {
            'shuffle': shuffle,
            'batch_size': batch_size,
            'num_workers': nworkers,
            'pin_memory': pin_memory
        }

    def get_loaders(self):
        return DataLoader(self.train_set, **self.init_kwargs), DataLoader(self.test_set, **self.init_kwargs)

class TinyImageNetDataLoader:

    def __init__(self, model_transform, batch_size=64, shuffle=True, nworkers=4, pin_memory=True, split= None):

        self.split = split

        if self.split:
            train_data,train_labels, val_data, val_labels = _Getdata(train=True, split = self.split)._get_data()
        else:
            train_data, train_labels = _Getdata(train=True, split = None)._get_data()
        
        test_data, test_labels = _Getdata(train=False, split= None)._get_data()
        
        self.train_set = TINDataset(
            train_data,
            train_labels,
            transform=model_transform.build_transforms(train=True)
        )

        if self.split:
            self.val_set = TINDataset(
            val_data,
            val_labels,
            transform=model_transform.build_transforms(train=False)
            )

        self.test_set = TINDataset(
            test_data,
            test_labels,
            transform=model_transform.build_transforms(train=False)
        )

        self.init_kwargs = {
            'shuffle': shuffle,
            'batch_size': batch_size,
            'num_workers': nworkers,
            'pin_memory': pin_memory
        }

    def get_loaders(self):

        if self.split:
            return DataLoader(self.train_set, **self.init_kwargs), DataLoader(self.val_set, **self.init_kwargs), DataLoader(self.test_set, **self.init_kwargs)
        else:
            return DataLoader(self.train_set, **self.init_kwargs), DataLoader(self.test_set, **self.init_kwargs)