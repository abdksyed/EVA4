import torch
import torch.nn.functional as F

test_loss = []

test_acc = []

max = [0]

def test(model, device, test_loader):
    model.eval()
    correct = 0
    tloss = 0

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
        path = '/classifier.pt'
        torch.save(model.state_dict(), path)