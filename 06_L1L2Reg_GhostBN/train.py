from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import model as m

train_loss = []

train_acc = []
train_endacc = []


def optimizer(model, lam2=0.0):
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=lam2)


def l1loss(model, lam):
    #l1_crit = nn.L1loss(size_average=False)
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.sum(param.abs())
    return lam*reg_loss


def train(model, device, train_loader, optimizer, epoch, lam=0.0):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        y_pred = model(data)

        loss = F.nll_loss(y_pred, target)
        if lam > 0:
            loss += l1loss(model, lam)

        train_loss.append(loss)

        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100 * correct/processed)
    train_endacc.append(train_acc[-1])
