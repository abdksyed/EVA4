from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Getting to know Data


def stat():
    data = datasets.MNIST(
        './data', train=True, transform=transforms.Compose([transforms.ToTensor(), ]), download=True)

    exp = data.data
    exp = data.transform(exp.numpy())

    print('Train Statistics')
    print(' - Numpy Shape:', data.data.cpu().numpy().shape)
    print(' - Tensor Shape:', data.data.size())
    print(' - min:', torch.min(exp))
    print(' - max:', torch.max(exp))
    print(' - mean:', torch.mean(exp))
    print(' - std:', torch.std(exp))

# Transforming Data (Normalizing to mean=1, std= 0)


def transform(mean, std, rot):
    if rot != 0.0:
        train_transform = transforms.Compose([
            transforms.RandomRotation((-rot, rot), fill=(1,)),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ]
        )
    elif rot == 0.0:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ]
        )

    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((mean,), (std,))
                                        ])

    return train_transform, test_transform

# Getting Train and Test Data


def split(mean=0.1311, std=0.3081, rot=0.0):
    train_transform, test_transform = transform(mean, std, rot)
    train = datasets.MNIST('./data', train=True,
                           transform=train_transform, download=True)
    test = datasets.MNIST('./data', train=False,
                          transform=test_transform, download=True)

    return train, test


def load(mean=0.1311, std=0.3081, rot=0.0):
    seed = 1

    train, test = split(mean, std, rot)

    # CUDA Availability
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For Reproducibility
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)

    dataloader_args = dict(shuffle=True, batch_size=64, num_workers=4,
                           pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # Train Dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    # Test Dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

    return train_loader, test_loader
