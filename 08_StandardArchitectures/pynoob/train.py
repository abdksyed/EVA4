import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Train:
    def __init__(self, model, device, train_loader, optimizer):
        
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.optimizer = optimizer

        self.train_loss = []

        self.train_acc = []
        self.train_endacc = []
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(data)

            loss = self.criterion(y_pred, target)

            self.train_loss.append(loss)

            loss.backward()
            self.optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100 * correct/processed)
        self.train_endacc.append(self.train_acc[-1])