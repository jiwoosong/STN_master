import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Classification_net(nn.Module):
    def __init__(self, in_chanel = 3,out_chanel = 43):
        super(Classification_net, self).__init__()
        self.name = os.path.realpath(__file__).split('/')[-1]
        self.conv1 = nn.Conv2d(in_chanel, 128, kernel_size=[1, 1], stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=[1, 1], stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=[1, 1], stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, out_chanel, kernel_size=[1, 1], stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(64, 43)
        return x