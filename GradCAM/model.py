import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, ni,no):
        super().__init__()
        self.conv = nn.Conv2d(ni, no, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(no)
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self,x):
        y = F.relu(self.conv(x))
        y = self.maxpool(self.bn(y))
        return y

class MalariaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(3,64)
        self.convblock2 = ConvBlock(64,64)
        self.convblock3 = ConvBlock(64,128)
        self.convblock4 = ConvBlock(128,128)
        self.convblock5 = ConvBlock(128,256)
        self.convblock6 = ConvBlock(256,64)
        self.linear1 = nn.Linear(256,32)
        self.linear2 = nn.Linear(32,2)
        
    def forward(self,x):
        y = self.convblock1(x)
        y = self.convblock2(y)
        y = self.convblock3(y)
        y = self.convblock4(y)
        y = self.convblock5(y)
        y = self.convblock6(y)
        y = torch.flatten(y,1)
        y = F.relu(self.linear1(y))
        return self.linear2(y)
    
    def im2fmap(self,x):
        z = self.convblock1(x)
        z = self.convblock2(z)
        z = self.convblock3(z)
        z = self.convblock4(z)
        z = self.convblock5(z)
        z = self.convblock6.conv(z)
        return z