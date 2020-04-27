from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Transforming Data (Normalizing to mean=1, std= 0)
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1311,), (0.3081,))
]
)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1311,), (0.3081,))
])
# Getting Train and Test Data
train = datasets.MNIST('./data', train=True, transform=train_transform)
test = datasets.MNIST('./data', train=False, transform=test_transform)

seed = 1

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

# Testing Normalized Data Statistics


def single():
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
