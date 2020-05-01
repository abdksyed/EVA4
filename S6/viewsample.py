from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import data

train, test = data.split()
train_loader, test_loader = data.load()


def single(mean=0.1311, std=0.3081):

    # Testing Normalized Data Statistics

    train_numpy = train.train_data
    train_numpy = train.transform(train_numpy.numpy())

    print('[Train]')
    print(' - Numpy Shape:', train_numpy.shape)
    print(' - Tensor Shape:', train.data.size())
    print(' - min:', torch.min(train_numpy))
    print(' - max:', torch.max(train_numpy))
    print(' - mean:', torch.mean(train_numpy))
    print(' - std:', torch.std(train_numpy))
    print(' - var:', torch.var(train_numpy))

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    print(images.shape)
    print(labels.shape)

    # Let's visualize some of the images

    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')


def multi():
    figure = plt.figure()
    num_of_images = 64
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    for index in range(0, num_of_images):
        plt.subplot(8, 8, index+1)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
