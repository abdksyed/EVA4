import torch
import QuizDNN as m
from torchsummary import summary

def summ(device):

    print(device)
    model = m.Net().to(device)
    summary(model, input_size=(3, 32, 32))
