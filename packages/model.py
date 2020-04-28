class Net(nn.Module):

    def __init__(self, bn=False, drop=0.0):

        super(Net, self).__init__()
        self.bn = bn
        self.drop = drop

        # Input Block - Input= 28
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            if self.bn:
                nn.BatchNorm2d(8),
            if self.drop != 0.0:
                nn.Dropout2d(self.drop),
        )  # Output= 28 / ReceptiveField= 3

        # Convolution Block - Input= 28
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            if self.bn:
                nn.BatchNorm2d(8),
            if self.drop != 0.0:
                nn.Dropout2d(self.drop),
        )  # Output= 28 / ReceptiveField= 5

        # Max Pooling and 1x1 Convolution
        self.mp1 = nn.Sequential(
            nn.MaxPool2d(2)
        )  # Output= 14 / ReceptiveField= 6

        # Convolution Block - Input= 14
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12,
                      kernel_size=3, bias=False),
            nn.ReLU(),
            if self.bn:
                nn.BatchNorm2d(12),
            if self.drop != 0.0:
                nn.Dropout2d(self.drop),
            # Output= 12 / ReceptiveField= 10
            nn.Conv2d(in_channels=12, out_channels=12,
                      kernel_size=3, bias=False),
            nn.ReLU(),
            if self.bn:
                nn.BatchNorm2d(12),
            if self.drop != 0.0:
                nn.Dropout2d(self.drop),
        )  # Output= 10 / ReceptiveField= 14

        # Convolution Block - Input= 10
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12,
                      kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            if self.bn:
                nn.BatchNorm2d(12),
            if self.drop != 0.0:
                nn.Dropout2d(self.drop),
            # Output= 8 / ReceptiveField= 18
            nn.Conv2d(in_channels=12, out_channels=16,
                      kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            if self.bn:
                nn.BatchNorm2d(16),
            if self.drop != 0.0:
                nn.Dropout2d(self.drop),
        )  # Output= 6 / ReceptiveField= 22

        # Convolution Block - Input= 6
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10,
                      kernel_size=3, bias=False),
            if self.drop != 0.0:
            nn.Dropout2d(self.drop),
        )  # Output= 4 / ReceptiveField= 26

        # GAP - Input 3
        self.gap = nn.AvgPool2d(4)
        # Output= 1 / ReceptiveField= 32

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.mp1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)
