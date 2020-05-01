from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

test_loss = []

test_acc = []

internalkey = ['']
max = [0]


def test(model, device, test_loader, key):
    model.eval()
    correct = 0
    tloss = 0
    if internalkey[0] != key:
        max[0] = 0
        internalkey[0] = key

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            tloss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    tloss /= len(test_loader.dataset)
    test_loss.append(tloss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          tloss, correct, len(test_loader.dataset),
          100 * correct/len(test_loader.dataset)))

    test_acc.append(100 * correct/len(test_loader.dataset))
    if test_acc[-1] > max[0]:
        max[0] = test_acc[-1]
        path = 'savedmodel/' + key + ' classifier.pt'
        torch.save(model.state_dict(), path)
