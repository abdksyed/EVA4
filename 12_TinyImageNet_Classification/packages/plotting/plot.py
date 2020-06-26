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
from torchvision.utils import save_image
from math import floor

from .gradCAM import GradCAM
from ..load_data import _Getdata

pred_list= []
true_list = []
classes = []
device = 'cuda'

def inv_norm(image, dataset_used= 'CIFAR10'):

    if dataset_used == 'CIFAR10':
      mean = (-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010)
      std = (1/0.2023, 1/0.1994, 1/0.2010)
    if dataset_used == 'TinyImageNet':
      mean = (-0.4802/0.2302, -0.4481/0.1994, -0.3975/0.2262)
      std = (1/0.2302, 1/0.2265, 1/0.2262)
    inv_norm_transform = transforms.Normalize(
        mean=mean,
        std=std)
    return inv_norm_transform(image)
    
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

def mis(model, device, test_loader, nimage = 64, dataset_used= 'CIFAR10'):
    """Display the 'nimage' number of misclassified images."""
    
    if dataset_used == 'CIFAR10':
      classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if dataset_used == 'TinyImageNet':
      classes = _Getdata()._get_class_to_id_dict()
      classes = list(classes.keys())

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
    #else:
        #print("Directory " , dirName, 'and', subdir ,  " already exists")
    ###############

    tlab, plab, img= _test_mis(model, device, test_loader)

    print(img.shape)
    plt.figure(figsize=(16,16))

    for index in range(0, nimage):
        plt.subplot(int(np.sqrt(nimage)), int(np.sqrt(nimage)), index+1)
        plt.xticks([])
        plt.yticks([])
        x = inv_norm(img[index], dataset_used= dataset_used)      # unnormalize
        x = x.permute(1, 2, 0) # (C, M, N) -> (M, N, C)
        x = x.cpu().numpy()

        ###########
        #To Save mis classified image for further use in GradCAM
        path = f'/content/mis_class/images/mis_{index+10}.png'
        mis_img = Image.fromarray((x.squeeze() * 255/x.max()).astype(np.uint8))
        mis_img.save(path)
        #########

        plt.imshow((x.squeeze() * 255/x.max()).astype(np.uint8), cmap='gray_r', interpolation= 'bilinear')
        plt.setp(plt.title(f'Predicted: {classes[plab[index,0]]}'), color= 'red')
        pred_list.append(classes[plab[index,0]])
        plt.setp(plt.xlabel(f'Ground Truth: {classes[tlab[index,0]]}'), color= 'blue')
        true_list.append(classes[tlab[index,0]])
        plt.tight_layout()

def _visualize_cam(mask, img, hm_lay=0.5, img_lay=0.5, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap*hm_lay + img.cpu()*img_lay
    result = result.div(result.max()).squeeze()

    return heatmap, result

def gen_cam(model, layer, class_idx= None, hm_lay= 0.5, img_lay= 0.5, dataset_used= 'CIFAR10'):
    
    #############
    #Create directory for saving GradCAM images.

    # Create target Directory if don't exist
    if class_idx is None:
        dirName = '/content/result_pred/'
        dir2Name = '/content/heatmap_pred/'
        if not os.path.exists(dirName) or not os.path.exists(dirName):
            os.mkdir(dirName)
            os.mkdir(dir2Name)
            print("Directory " , dirName ,  " Created ")
            print("Directory " , dir2Name ,  " Created ")
        #else:
            #print("Directory " , dirName , " already exists")
            #print("Directory " , dir2Name , " already exists")
    else:
        dirName = '/content/result_act/'
        dir2Name = '/content/heatmap_act/'
        if not os.path.exists(dirName) or not os.path.exists(dirName):
            os.mkdir(dirName)
            os.mkdir(dir2Name)
            print("Directory " , dirName ,  " Created ")
            print("Directory " , dir2Name ,  " Created ")
        #else:
           #print("Directory " , dirName , " already exists")
           #print("Directory " , dir2Name , " already exists")
    
    if dataset_used == 'CIFAR10':
      classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if dataset_used == 'TinyImageNet':
      classes = _Getdata()._get_class_to_id_dict()
      classes = list(classes.keys())
      
    model_layer = getattr(model, layer)
    gradcam = GradCAM(model, model_layer)

    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    dataset = ImageFolder(root='/content/mis_class/', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    
    for index, batch in enumerate(dataloader):
      normed_torch_img, _ = batch

      torch_img = inv_norm(normed_torch_img[0], dataset_used=dataset_used)
      if class_idx is None:
          class_idx_ = None
      else:
          class_idx_ = classes.index(class_idx[index])
      mask, _ = gradcam(normed_torch_img.to(device), class_idx = class_idx_)
      heatmap, result = _visualize_cam(mask, torch_img, hm_lay, img_lay)
      if class_idx is None:
        fp_path_heat = f'/content/heatmap_pred/map{index+10}_{layer}.png'
        save_image(heatmap, fp=fp_path_heat)
        fp_path_res = f'/content/result_pred/result{index+10}_{layer}.png'
        save_image(result, fp=fp_path_res)
      else:
        fp_path_heat = f'/content/heatmap_act/map{index+10}_{layer}.png'
        save_image(heatmap, fp=fp_path_heat)
        fp_path_res = f'/content/result_act/result{index+10}_{layer}.png'
        save_image(result, fp=fp_path_res)

def plot_pred_cam(n,l):
  fig, axes = plt.subplots(n, l+2, figsize=((l+2)*3,n*2.5))
  fig.suptitle("Grad-CAM of Mis Classified Images with respect to Predicted(wrong) Class", fontsize=20)
  for index in range(0, n):
      # plt.subplot(n, l+1, index+1)
      plt.xticks([])
      plt.yticks([])
      axes[index, 0].text(0.5, 0.5, f'Predicted {pred_list[index]} \n Actual: {true_list[index]}', fontsize= 16)
      axes[index, 0].axis('off')
      path = '/content/mis_class/images/mis_' + str(index+10) + '.png'
      img = plt.imread(path)
      axes[index, 1].imshow(img, interpolation= 'bilinear')
      axes[index, 1].axis('off')
      for layer in range(l):
        path = f'/content/result_pred/result{index+10}_layer{layer+1}.png'
        img = plt.imread(path)
        axes[index, layer+2].imshow(img.squeeze(), cmap='gray_r', interpolation= 'bilinear')
        axes[index, layer+2].set_title(f'Layer: {layer+1}')
        axes[index, layer+2].axis('off')
  plt.tight_layout()
  plt.subplots_adjust(top=0.97)
  plt.show()

def plot_act_cam(n,l):
  fig, axes = plt.subplots(n, l+2, figsize=((l+2)*3,n*2.5))
  fig.suptitle("Grad-CAM of Mis Classified Images with respect to Actual(correct) Class", fontsize=20)
  for index in range(0, n):
      # plt.subplot(n, l+1, index+1)
      plt.xticks([])
      plt.yticks([])
      axes[index, 0].text(0.5, 0.5, f'Predicted {pred_list[index]} \n Actual: {true_list[index]}', fontsize= 16)
      axes[index, 0].axis('off')
      path = '/content/mis_class/images/mis_' + str(index+10) + '.png'
      img = plt.imread(path)
      axes[index, 1].imshow(img, interpolation= 'bilinear')
      axes[index, 1].axis('off')
      for layer in range(l):
        path = f'/content/result_act/result{index+10}_layer{layer+1}.png'
        img = plt.imread(path)
        axes[index, layer+2].imshow(img.squeeze(), cmap='gray_r', interpolation= 'bilinear')
        axes[index, layer+2].set_title(f'Layer: {layer+1}')
        axes[index, layer+2].axis('off')
  plt.tight_layout()
  plt.subplots_adjust(top=0.97)
  plt.show()

def plot_pred_heatmap(n,l):
  fig, axes = plt.subplots(n, l+2, figsize=((l+2)*3,n*2.5))
  fig.suptitle("Grad-CAM - Heatmap of Mis Classified Images with respect to Predicted(wrong) Class", fontsize=20)
  for index in range(0, n):
      # plt.subplot(n, l+1, index+1)
      plt.xticks([])
      plt.yticks([])
      axes[index, 0].text(0.5, 0.5, f'Predicted {pred_list[index]} \n Actual: {true_list[index]}', fontsize= 16)
      axes[index, 0].axis('off')
      path = '/content/mis_class/images/mis_' + str(index+10) + '.png'
      img = plt.imread(path)
      axes[index, 1].imshow(img, interpolation= 'bilinear')
      axes[index, 1].axis('off')
      for layer in range(l):
        path = f'/content/heatmap_pred/heatmap{index+10}_layer{layer+1}.png'
        img = plt.imread(path)
        axes[index, layer+2].imshow(img.squeeze(), cmap='gray_r', interpolation= 'bilinear')
        axes[index, layer+2].set_title(f'Layer: {layer+1}')
        axes[index, layer+2].axis('off')
  plt.tight_layout()
  plt.subplots_adjust(top=0.97)
  plt.show()

def plot_act_heatmap(n,l):
  fig, axes = plt.subplots(n, l+2, figsize=((l+2)*3,n*2.5))
  fig.suptitle("Grad-CAM - Heatmap of Mis Classified Images with respect to Actual(correct) Class", fontsize=20)
  for index in range(0, n):
      # plt.subplot(n, l+1, index+1)
      plt.xticks([])
      plt.yticks([])
      axes[index, 0].text(0.5, 0.5, f'Predicted {pred_list[index]} \n Actual: {true_list[index]}', fontsize= 16)
      axes[index, 0].axis('off')
      path = '/content/mis_class/images/mis_' + str(index+10) + '.png'
      img = plt.imread(path)
      axes[index, 1].imshow(img, interpolation= 'bilinear')
      axes[index, 1].axis('off')
      for layer in range(l):
        path = f'/content/heatmap_act/heatmap{index+10}_layer{layer+1}.png'
        img = plt.imread(path)
        axes[index, layer+2].imshow(img.squeeze(), cmap='gray_r', interpolation= 'bilinear')
        axes[index, layer+2].set_title(f'Layer: {layer+1}')
        axes[index, layer+2].axis('off')
  plt.tight_layout()
  plt.subplots_adjust(top=0.97)
  plt.show()


def plot_cycle(no_cycle, loader, step_size=None, min_lr= 0.0001, max_lr = 10):

  if step_size is None:
    step_size = 8*len(loader)

  total_iter = no_cycle*(2*step_size)

  lr_list = [min_lr]

  for i in range(1, total_iter+1):
    cycle = floor(1 + i/(2*step_size))
    x = abs((i/step_size) - (2*cycle) + 1)
    curr_lr = min_lr + (max_lr - min_lr)*(1-x)
    lr_list.append(curr_lr)

  plt.figure(figsize= (no_cycle, 3))
  plt.plot(lr_list)
  plt.title('Cyclic Learning Rate')
  plt.xlabel('Iterations')
  plt.ylabel('Learning Rate')