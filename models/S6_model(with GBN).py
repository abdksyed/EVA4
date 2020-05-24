from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(
            num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(
            num_features * self.num_splits))

    def train(self, mode=True):
        # lazily collate stats when we are going to use them
        if (self.training is True) and (mode is False):
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H,
                           W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(
                    self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


def reg(features, num_splits, gbn):
    if gbn:
        return GhostBatchNorm(features, num_splits)
    else:
        return nn.BatchNorm2d(features)


class Net(nn.Module):

    def __init__(self, gbn):

        super(Net, self).__init__()
        self.gbn = gbn

        # Input Block - Input= 28
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=3, padding=1, bias=False),
            reg(8, 4, self.gbn),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
        )  # Output= 28 / ReceptiveField= 3

        # Convolution Block - Input= 28
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8,
                      kernel_size=3, padding=1, bias=False),
            reg(8, 4, self.gbn),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
        )  # Output= 28 / ReceptiveField= 5

        # Max Pooling and 1x1 Convolution
        self.mp1 = nn.Sequential(
            nn.MaxPool2d(2)
        )  # Output= 14 / ReceptiveField= 6

        # Convolution Block - Input= 14
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12,
                      kernel_size=3, bias=False),
            reg(12, 4, self.gbn),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            # Output= 12 / ReceptiveField= 10
            nn.Conv2d(in_channels=12, out_channels=12,
                      kernel_size=3, bias=False),
            reg(12, 4, self.gbn),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
        )  # Output= 10 / ReceptiveField= 14

        # Convolution Block - Input= 10
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12,
                      kernel_size=3, padding=0, bias=False),
            reg(12, 4, self.gbn),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            # Output= 8 / ReceptiveField= 18
            nn.Conv2d(in_channels=12, out_channels=16,
                      kernel_size=3, padding=0, bias=False),
            reg(16, 4, self.gbn),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
        )  # Output= 6 / ReceptiveField= 22

        # Convolution Block - Input= 6
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, bias=False),
            reg(16, 4, self.gbn),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
        )  # Output= 4 / ReceptiveField= 26

        # GAP - Input 3
        self.gap = nn.AvgPool2d(4)
        # Output= 1 / ReceptiveField= 26

        self.conv = nn.Conv2d(
            in_channels=16, out_channels=10, kernel_size=1, bias=False)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.mp1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = self.conv(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)
