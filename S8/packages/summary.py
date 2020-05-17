import torch
import models.Net as m
from torchsummary import summary

def summ(device, model_name):

    model_ = getattr(m, model_name)
    model_ = model_().to(device)
    summary(model_, input_size=(3, 32, 32))