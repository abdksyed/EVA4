import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import math

class Range_Test():

    def __init__(self, device, epochs, model_name, loader, start_lr=1e-4, end_lr=10,
                 criterion= nn.CrossEntropyLoss(), optimizer_name= 'SGD', smoothing= 0.05):

        self.stats = {'lr': [], 'loss': [], 'acc': []}
        self.device = device
        self.criterion = criterion
        self.model = model_name
        self.optimizer = getattr(torch.optim, optimizer_name)
        self.optimizer = self.optimizer(self.model.parameters(), lr=start_lr, momentum= 0.9)
        self.loader = loader
        self.epochs = epochs
        self.smoothing = smoothing
        
        lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (epochs * len(loader)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def range_test(self):
    
        print('='*20 + f' START ' + '='*20)

        for epoch in range(self.epochs):
        
            print('='*20 + f' EPOCH: {epoch+1} ' + '='*20)
            pbar = tqdm(self.loader)

            for batch_idx, (data, target) in enumerate(pbar):
            
                self.model.train()

                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                lr_step = self.optimizer.param_groups[0]["lr"]
                self.stats['lr'].append(lr_step)

                y_pred = self.model(data)

                loss = self.criterion(y_pred, target)

                loss.backward()
                self.optimizer.step()

                #Making Prediction for Accuracy
                pred = y_pred.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                processed = len(data)
                self.stats['acc'].append((correct/processed)*100)

                # Update LR
                self.scheduler.step()                

                # smooth the loss
                if self.stats['loss'] == []:
                  self.stats['loss'].append(loss)
                else:
                  loss = self.smoothing  * loss + (1 - self.smoothing) * self.stats['loss'][-1]
                  self.stats['loss'].append(loss)

    def plot(self):

        fig, axis = plt.subplots(1,2, figsize= (10,4))
        axis[0].plot(self.stats['lr'], self.stats['loss'])
        axis[0].set_xlabel('Learning Rate')
        axis[0].set_ylabel('Loss')
        axis[0].set_title('Range Test - LR vs Loss')
        self.lr_loss = self.stats['lr'][self.stats['loss'].index(min(self.stats['loss']))]
        axis[0].axvline(x= self.lr_loss, color= 'red')
        axis[0].set_xscale('log')

        axis[1].plot(self.stats['lr'], self.stats['acc'])
        axis[1].set_xlabel('Learning Rate')
        axis[1].set_ylabel('Accuracy')
        axis[1].set_title('Range Test - LR vs Accuracy')
        self.lr_acc = self.stats['lr'][self.stats['acc'].index(max(self.stats['acc']))]
        axis[1].axvline(x= self.lr_acc, color= 'red')
        axis[1].set_xscale('log')

        print('='*49)
        print(f'The LR for Least Loss is: {self.lr_loss}')
        print(f'The LR for Maximum Accuracy is: {self.lr_acc}')
        print('='*49)