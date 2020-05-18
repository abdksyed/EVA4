import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For Reproducibility
torch.manual_seed(1)

if cuda:
    torch.cuda.manual_seed(1)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def loader(batch_size= 64):
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4,
                            pin_memory=True) if cuda else dict(shuffle=True, batch_size=32)

    train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

    return train_loader, test_loader

def _imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #print(npimg.shape)
    #print(np.transpose(npimg, (1, 2, 0)).shape)
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def display(n= 64):
    
    train_loader, test_loader = loader()

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    for i in range(0,n,int(np.sqrt(n))):
        #print('the incoming image is ',images[i: i+int(np.sqrt(n))].shape)
        _imshow(torchvision.utils.make_grid(images[i: i+int(np.sqrt(n))]))
        # print labels
        plt.title(' '.join('%7s' % classes[j] for j in labels[i: i+int(np.sqrt(n))]), loc= 'left')
