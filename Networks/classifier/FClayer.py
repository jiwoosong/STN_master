import torch
import torch.nn as nn
import os
import numpy as np

class Classification_net(nn.Module):
    def __init__(self,W_warp=50,H_warp=50,chanel=3):
        super(Classification_net, self).__init__()
        self.name = os.path.realpath(__file__).split('/')[-1]
        self.fc1 = nn.Linear(W_warp*H_warp*chanel, 128)
        self.fc2 = nn.Linear(128, 43)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x= x.reshape(batch_size, -1)
        x = self.relu(self.fc1(x)) # FC1 (128)
        x = self.relu(self.fc2(x)) # FC2 (classes=43)
        return x
