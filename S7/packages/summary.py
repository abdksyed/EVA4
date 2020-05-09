import torch
import model as m
from torchsummary import summary

def summ(device):

    print(device)
    model = m.Net(False).to(device)
    summary(model, input_size=(3, 32, 32))