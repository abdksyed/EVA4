import torch
import resnet as m
from torchsummary import summary

def summ(device):

    print(device)
    model = m.ResNet18().to(device)
    summary(model, input_size=(3, 32, 32))