import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import torchvision

from .datasets import _Getdata

def model_summary(model, input_size= (3, 32, 32)):
    """Displays the Summary of the Architecture - All the methods used and the parameters used"""
    summary(model, input_size=input_size)

def _imshow(img):
    inv_norm = transforms.Normalize(
        mean=(-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010),
        std=(1/0.2023, 1/0.1994, 1/0.2010))
    img = inv_norm(img)      # unnormalize
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation= 'bilinear')

def display(train_loader, n= 64, dataset_used = 'CIFAR10'):

    if dataset_used == 'CIFAR10':
      classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if dataset_used == 'TinyImageNet':
      classes = _Getdata()._get_class_to_id_dict()
      classes = list(classes.keys())

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    for i in range(0,n,int(np.sqrt(n))):
        _imshow(torchvision.utils.make_grid(images[i: i+int(np.sqrt(n))]))
        # print labels
        plt.title(' '.join('%7s' % classes[j] for j in labels[i: i+int(np.sqrt(n))]), loc= 'left')