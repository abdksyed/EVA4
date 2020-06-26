import torch
import torch.nn as nn
import torch.nn.functional as F

class Test:
    def __init__(self, model, device, test_loader, criterion):
        self.model = model
        self.device = device
        self.test_loader = test_loader

        self.test_loss = []
        self.test_acc = []
        self.criterion = criterion
        self.max_val = 0

    def test(self):
        self.model.eval()
        correct = 0
        tloss = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                tloss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        tloss /= len(self.test_loader.dataset)
        self.test_loss.append(tloss)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            tloss, correct, len(self.test_loader.dataset),
            100 * correct/len(self.test_loader.dataset)))

        self.test_acc.append(100 * correct/len(self.test_loader.dataset))

        if self.test_acc[-1] > self.max_val:
            self.max_val = self.test_acc[-1]
            path = '/content/classifier.pt'
            torch.save(self.model.state_dict(), path)