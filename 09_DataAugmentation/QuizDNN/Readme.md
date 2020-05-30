
Quiz File

**Members:**

Jahnavi Ramagiri

Sachin Sharma

Madalasa Venkataraman

Syed Abdul Khader

Colab File : https://colab.research.google.com/drive/1R3QyZlijXn0UzxfLeQGEkq2MAI_cYKSW?usp=sharing

**Objective**

Make a DNN such that:

it&#39;s first block uses following code:

import datetime from datetime

print(&quot;Current Date/Time: &quot;, datetime.now())

uses the Modules you have written

calls following DNN from a file called QuizDNN.py:

x1 = Input

x2 = Conv(x1)

x3 = Conv(x1 + x2)

x4 = MaxPooling(x1 + x2 + x3)

x5 = Conv(x4)

x6 = Conv(x4 + x5)

x7 = Conv(x4 + x5 + x6)

x8 = MaxPooling(x5 + x6 + x7)

x9 = Conv(x8)

x10 = Conv (x8 + x9)

x11 = Conv (x8 + x9 + x10)

x12 = GAP(x11)

x13 = FC(x12)

Uses ReLU and BN wherever applicable

Uses CIFAR10 as the dataset

Your target is 75% in less than 40 Epochs

Once Done:

Paste the code in QuizDNN.py

Paste the code of your Colab (or computer&#39;s) Notebook

Paste the complete training log

Paste the link to your Google Colab Notebook file (or you GitHub Repo)

**Our Network**

QuizDNN.py code

import torch.nn as nn

import torch.nn.functional as F

class Net(nn.Module):

def \_\_init\_\_(self):

super(Net, self).\_\_init\_\_()

self.convblock1 = nn.Sequential(

nn.Conv2d(in\_channels=3, out\_channels=64, kernel\_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1

nn.ReLU(),

nn.BatchNorm2d(64),

nn.Dropout(p = 0.1),

)

self.convblock2 = nn.Sequential(

nn.Conv2d(in\_channels=64, out\_channels=64, kernel\_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1

nn.ReLU(),

nn.BatchNorm2d(64),

nn.Dropout(p = 0.1),

)

self.convblock3 = nn.Sequential(

nn.Conv2d(in\_channels=64, out\_channels=64, kernel\_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1

nn.ReLU(),

nn.BatchNorm2d(64),

nn.Dropout(p = 0.1),

)

self.convblock4 = nn.Sequential(

nn.Conv2d(in\_channels=64, out\_channels=64, kernel\_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1

nn.ReLU(),

nn.BatchNorm2d(64),

nn.Dropout(p = 0.1),

)

self.convblock5 = nn.Sequential(

nn.Conv2d(in\_channels=64, out\_channels=64, kernel\_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1

nn.ReLU(),

nn.BatchNorm2d(64),

nn.Dropout(p = 0.1),

)

self.convblock6 = nn.Sequential(

nn.Conv2d(in\_channels=64, out\_channels=64, kernel\_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1

nn.ReLU(),

nn.BatchNorm2d(64),

nn.Dropout(p = 0.1),

)

self.convblock7 = nn.Sequential(

nn.Conv2d(in\_channels=64, out\_channels=64, kernel\_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1

nn.ReLU(),

nn.BatchNorm2d(64),

nn.Dropout(p = 0.1),

)

self.convblock8 = nn.Sequential(

nn.Conv2d(in\_channels=64, out\_channels=64, kernel\_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1

nn.ReLU(),

nn.BatchNorm2d(64),

nn.Dropout(p = 0.1),

)

self.convblock9 = nn.Sequential(

nn.Conv2d(in\_channels=64, out\_channels=64, kernel\_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1

nn.ReLU(),

nn.BatchNorm2d(64),

nn.Dropout(p = 0.1),

)

self.pool = nn.MaxPool2d(2, 2)

self.gap = nn.Sequential(

nn.AdaptiveAvgPool2d(1)

)

self.fc = nn.Conv2d(in\_channels=64, out\_channels=10, kernel\_size=(1, 1), padding=0, bias=False)#Op\_size = 1,

def forward(self, x):

x1 = self.convblock1(x)

x2 = self.convblock2(x1)

x3 = self.convblock3(x1+x2)

x4 = self.pool(x1+x2+x3)

x5 = self.convblock4(x4)

x6 = self.convblock5(x4+x5)

x7 = self.convblock6(x4+x5+x6)

x8 = self.pool(x5+x6+x7)

x9 = self.convblock7(x8)

x10 = self.convblock8(x8+x9)

x11 = self.convblock9(x8+x9+x10)

x12 = self.gap(x11)

x13 = self.fc(x12)

x = x13.view(-1, 10)

return x

**Packages**

used modules for data.py, train.py, test.py, Albtransform.py, summary2.py, utils,classifier.py
