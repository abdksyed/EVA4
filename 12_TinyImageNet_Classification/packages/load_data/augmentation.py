import abc

import cv2
import torchvision.transforms as T
import albumentations as alb
from albumentations.pytorch import ToTensor
import numpy as np

class AlbumentationTrans:
    
    def __init__(self, transform):
        self.album_transform = transform

    def __call__(self, img):
        img = np.array(img)
        return self.album_transform(image=img)['image']

class AugmentationBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class MNIST_Transforms(AugmentationBase):

    def build_train(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))])

    def build_test(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))])


class CIFAR10_Transforms(AugmentationBase):

    def build_train(self):
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def build_test(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

class CIFAR10_AlbumTrans(AugmentationBase):

    def __init__(self, augs= None):
      self.augs = augs

    def build_train(self):
        
        if self.augs is None:
          train_trans = alb.Compose([
                                     
              alb.PadIfNeeded(40,40,cv2.BORDER_CONSTANT,[4,4],0),
              alb.RandomCrop(32,32, always_apply= True),
              alb.HorizontalFlip(p= 0.75),
              alb.CoarseDropout(
                  max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, 
                  fill_value=[0.4914*255, 0.4822*255, 0.4465*255], always_apply=False, p=0.75
              ),
              alb.Normalize(
                  mean=[0.4914, 0.4822, 0.4465],
                  std=[0.2023, 0.1994, 0.2010]
              ),
              ToTensor()
          ])
        else:
          train_trans = alb.Compose(self.augs)
        
        return AlbumentationTrans(train_trans)

    def build_test(self):
        test_trans = alb.Compose([
            alb.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            ToTensor()
        ])
        return AlbumentationTrans(test_trans)

class TinyImageNet_AlbumTrans(AugmentationBase):

    def __init__(self, augs= None):
      self.augs = augs

    def build_train(self):
        
        if self.augs is None:
          train_trans = alb.Compose([
            
              alb.RandomCrop(56,56, always_apply= True),
              alb.HorizontalFlip(),
              alb.Rotate((-20,20)),
              alb.CoarseDropout(
                  max_holes=1, max_height=28, max_width=28, min_holes=1, min_height=28, min_width=28, 
                  fill_value=[0.4802*255, 0.4481*255, 0.3975*255], always_apply=False
              ),
              alb.Normalize(
                  mean = [0.4802, 0.4481, 0.3975],
                  std = [0.2302, 0.2265, 0.2262]
              ),
              ToTensor()
          ])
        else:
          train_trans = alb.Compose(self.augs)
        
        return AlbumentationTrans(train_trans)

    def build_test(self):
        test_trans = alb.Compose([
            alb.Resize(56,56),
            alb.Normalize(
                mean = [0.4802, 0.4481, 0.3975],
                std = [0.2302, 0.2265, 0.2262]
            ),
            ToTensor()
        ])
        return AlbumentationTrans(test_trans)