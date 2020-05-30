import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.1),
            
            )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.1),
            
            )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.1),
            
            )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.1),
            
            )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.1),
            
            )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.1),
            
            )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.1),
            
            )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.1),
            
            )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.1),
            
            )

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
            

        self.fc =  nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)#Op_size = 1, 

      


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
