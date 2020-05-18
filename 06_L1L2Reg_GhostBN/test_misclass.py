from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

total_loss = []


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    true_label = torch.Tensor([]).to(device)
    true_label = true_label.long()
    pred_label = torch.Tensor([]).to(device)
    pred_label = pred_label.long()
    misclass_image = torch.Tensor([]).to(device)

    with torch.no_grad():
        for data, target in test_loader:
            img_batch = data
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            #result = pred.eq(target.view_as(pred))

            # Save incorrect samples

            #print("Actual label:\n",target.shape)
            #print("Predicted output:\n", pred.shape)
            #print("Shape of data:\n",data.shape)

            # To check if the predicted output is equal to the label.
            index_wrong = ~pred.eq(target.view_as(pred))

            true_label = torch.cat(
                (true_label, target.view_as(pred)[index_wrong]), dim=0)
            pred_label = torch.cat((pred_label, pred[index_wrong]), dim=0)
            misclass_image = torch.cat(
                (misclass_image, data[index_wrong]), dim=0)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    total_loss.append(test_loss)
    # reg_accuracy.append
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), (100. * correct / len(test_loader.dataset))))
    return true_label, pred_label, misclass_image
