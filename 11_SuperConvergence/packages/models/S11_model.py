'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        #out = out + x  #Can be used to provide extra shortcut inside ResBlock itself

        out = F.relu(out)

        return out


class ParallelBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ParallelBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, padding=1, bias=False)
        self.maxp = nn.MaxPool2d(2)
        self.bn = nn.BatchNorm2d(out_channels)

        self.res = BasicResBlock(out_channels, out_channels)

    def forward(self, x):

        out = self.conv(x)
        out = self.maxp(out)
        out = self.bn(out)
        out = F.relu(out)
        out = out + self.res(out)

        return out


class S11_Model(nn.Module):

    def __init__(self):
        super(S11_Model, self).__init__()

        # Preperation Block - Input 32x32x3
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Output - 32x32x64

        # Input - 32x32x64
        self.layer1 = ParallelBlock(64, 128)
        # Output - 16x16x128

        # Input - 16x16x128
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # Output - 8x8x256

        # Input - 8x8x256
        self.layer3 = ParallelBlock(256, 512)
        # Output - 4x4x512

        self.max_k4 = nn.MaxPool2d(4)
        # Output - 1x1x512

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.max_k4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

