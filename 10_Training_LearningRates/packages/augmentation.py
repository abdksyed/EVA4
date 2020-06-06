import abc

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

    def build_train(self):
        train_trans = alb.Compose([
            alb.Rotate((-7.0, 7.0)),
            #alb.Cutout(1,16,16, [0.4914, 0.4822, 0.4465]*255),
            alb.CoarseDropout(1, 16, 16, fill_value= [0.4914*255, 0.4822*255, 0.4465*255]),
            #alb.RandomCrop(224,224),
            alb.HorizontalFlip(),
            alb.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
            ToTensor()
        ])
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

