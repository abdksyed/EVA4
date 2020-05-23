from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from .augmentation import *

class MNISTDataLoader:
    """
    It creates a data loader for test and train. It taken transformations from the 'augmentation' module
    """
    def __init__(self, model_transform, data_dir= './root', batch_size=64, shuffle=True, nworkers=4, pin_memory=True):
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

    def __init__(self, model_transform, data_dir= './root', batch_size=64, shuffle=True, nworkers=4, pin_memory=True):
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