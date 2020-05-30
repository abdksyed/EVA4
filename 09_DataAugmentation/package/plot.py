import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
from PIL import Image
import cv2
from torchvision.datasets import ImageFolder

from grad_model import ResNet_Grad

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
pred_list= []
true_list = []

def _test_mis(model, device, test_loader):
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
    """Display the 'nimage' number of misclassified images."""
    
    #############
    #Create directory for saving misclassified images 
    dirName = '/content/mis_class/'
    subdir = '/content/mis_class/images'

    # Create target Directory if don't exist
    if not os.path.exists(dirName) and not os.path.exists(subdir):
        os.mkdir(dirName)
        os.mkdir(subdir)
        print("Directory " , dirName ,  " Created ")
        print("Directory " , subdir ,  " Created ")
    elif not os.path.exists(subdir):
        os.mkdir(subdir)
        print("Directory " , subdir ,  " Created ")
    else:
        print("Directory " , dirName, 'and', subdir ,  " already exists")
    ###############

    tlab, plab, img= _test_mis(model, device, test_loader)

    inv_norm = transforms.Normalize(
        mean=(-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010),
        std=(1/0.2023, 1/0.1994, 1/0.2010))
    print(img.shape)
    
    figure = plt.figure()
    plt.figure(figsize=(16,16))

    for index in range(0, nimage):
        plt.subplot(int(np.sqrt(nimage)), int(np.sqrt(nimage)), index+1)
        plt.xticks([])
        plt.yticks([])
        x = inv_norm(img[index])      # unnormalize
        x = x.permute(1, 2, 0) # (C, M, N) -> (M, N, C)
        x = x.cpu().numpy()

        ###########
        #To Save mis classified image for further use in GradCAM
        path = '/content/mis_class/images/mis_' + str(index) + '.png'
        mis_img = Image.fromarray((x.squeeze() * 255).astype(np.uint8))
        mis_img.save(path)
        #########

        plt.imshow(x.squeeze(), cmap='gray_r')
        plt.setp(plt.title(f'Predicted: {classes[plab[index,0]]}'), color= 'red')
        pred_list.append(classes[plab[index,0]])
        plt.setp(plt.xlabel(f'Ground Truth: {classes[tlab[index,0]]}'), color= 'blue')
        true_list.append(classes[tlab[index,0]])
        plt.tight_layout()

def graph(train_obj, test_obj):
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

def gen_grad(model, nimage=64, map_overlay= 0.3, img_overlay= 0.7):
  
  #############
  #Create directory for saving GradCAM images.
  dirName = '/content/map/'
  dir2Name = '/content/heatmap/'

  # Create target Directory if don't exist
  if not os.path.exists(dirName) or not os.path.exists(dirName):
      os.mkdir(dirName)
      os.mkdir(dir2Name)
      print("Directory " , dirName ,  " Created ")
      print("Directory " , dir2Name ,  " Created ")
  else:
      print("Directory " , dirName , " already exists")
      print("Directory " , dir2Name , " already exists")
  ###############

  transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
  dataset = ImageFolder(root='/content/mis_class/', transform=transform)
  dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

  for index, batch in enumerate(dataloader):

        #getting the image from loader
        mis_img, _ = batch
        mis_img = mis_img.to('cuda')
        
        # init the resnet
        resnet = ResNet_Grad(model, mis_img)

        # set the evaluation mode
        _ = resnet.eval()


        # forward pass
        pred = resnet(mis_img)

        #prd_out = pred.argmax(dim=1) # prints tensor([2])
        #prd_out = prd_out.item()

        # get the gradient of the output with respect to the parameters of the model
        pred[:, classes.index(true_list[index])].backward()

        # pull the gradients out of the model
        gradients = resnet.get_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = resnet.get_activations(mis_img).detach()

        # weight the channels by corresponding gradients
        for i in range(512):
          activations[:, i, :, :] *= pooled_gradients[i]
      
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap.cpu(), 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        #plt.matshow(heatmap.squeeze())

        # make the heatmap to be a numpy array
        heatmap = heatmap.numpy()

        # interpolate the heatmap
        cvpath = '/content/mis_class/images/mis_' + str(index) + '.png'
        img = cv2.imread(cvpath)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_path = '/content/heatmap/heatmap_' + str(index) + '.png'
        cv2.imwrite(heatmap_path, heatmap.squeeze()) 
        superimposed_img = heatmap * map_overlay + img * img_overlay
        map_path = '/content/map/map_' + str(index) + '.png'
        cv2.imwrite(map_path, superimposed_img)

def plt_grad():
    figure = plt.figure()
    plt.figure(figsize=(16,16))
    for index in range(0, 36):
        plt.subplot(6, 6, index+1)
        plt.xticks([])
        plt.yticks([])
        path = '/content/map/map_' + str(index) + '.png'
        xa = plt.imread(path)
        plt.imshow(xa.squeeze(), cmap='gray_r')
        plt.setp(plt.title(f'Predicted: {pred_list[index]}'), color= 'red')
        plt.setp(plt.xlabel(f'Ground Truth: {true_list[index]}'), color= 'blue')
        plt.tight_layout()

def heat_map():
    figure = plt.figure()
    plt.figure(figsize=(16,16))
    for index in range(0, 36):
        plt.subplot(6, 6, index+1)
        plt.xticks([])
        plt.yticks([])
        path = '/content/heatmap/heatmap_' + str(index) + '.png'
        xa = plt.imread(path)
        plt.imshow(xa.squeeze(), cmap='gray_r')
        plt.setp(plt.title(f'Predicted: {pred_list[index]}'), color= 'red')
        plt.setp(plt.xlabel(f'Ground Truth: {true_list[index]}'), color= 'blue')
        plt.tight_layout()