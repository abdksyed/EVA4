import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import test, train

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

def test_mis(model, device, test_loader):
    model.eval()
    correct = 0
    tloss = 0

    true_label = torch.Tensor([]).to(device)
    true_label = true_label.long()
    pred_label = torch.Tensor([]).to(device)
    pred_label = pred_label.long()
    misclass_image = torch.Tensor([]).to(device)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            tloss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            index_wrong = ~pred.eq(target.view_as(pred))[:,0]
            true_label = torch.cat((true_label, target.view_as(pred)[index_wrong]), dim=0)
            pred_label = torch.cat((pred_label, pred[index_wrong]), dim=0)
            misclass_image = torch.cat((misclass_image, data[index_wrong]), dim=0)

    tloss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          tloss, correct, len(test_loader.dataset),
          100 * correct/len(test_loader.dataset)))
          
    return true_label, pred_label, misclass_image

def mis(model, device, test_loader, nimage = 64):
    
    tlab, plab, img= test_mis(model, device, test_loader)

    x = img.cpu().numpy()
    x = np.transpose(x, (0,2,3,1))
    x = x/2 + 0.5
    print(x.shape)

    figure = plt.figure()
    num_of_images = nimage
    plt.figure(figsize=(16,16))

    for index in range(0, num_of_images):
        plt.subplot(int(np.sqrt(nimage)), int(np.sqrt(nimage)), index+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x[index].squeeze(), cmap='gray_r')
        plt.title(f'Predicted: {classes[plab[index,0]]}')
        plt.xlabel(f'Ground Truth: {classes[tlab[index,0]]}')
        plt.tight_layout()

def graph():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train.train_loss)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train.train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test.test_loss)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test.test_acc)
    axs[1, 1].set_title("Test Accuracy")

def testvtrain():
    plt.axes(xlabel= 'epochs', ylabel= 'Accuracy')
    plt.plot(train.train_endacc)
    plt.plot(test.test_acc)
    plt.title('Test vs Train Accuracy')
    plt.legend(['Train', 'Test'])

def class_acc(model,device, test_loader):

    with torch.no_grad():
        for data, target in test_loader:
            images, labels = data.to(device), target.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))