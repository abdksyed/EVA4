import torch
import matplotlib.pyplot as plt

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def acc_loss(train_obj, test_obj):
    """Display the Train Loss and Accuracy graph. Test Loss and Accuracy graph."""
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_obj.train_loss)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_obj.train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_obj.test_loss)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_obj.test_acc)
    axs[1, 1].set_title("Test Accuracy")

def testvtrain(train_obj, test_obj):
    """Display Test vs Train Accuracy plot"""
    plt.axes(xlabel= 'epochs', ylabel= 'Accuracy')
    plt.plot(train_obj.train_endacc)
    plt.plot(test_obj.test_acc)
    plt.title('Test vs Train Accuracy')
    plt.legend(['Train', 'Test'])

def class_acc(model,device, test_loader):            
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

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