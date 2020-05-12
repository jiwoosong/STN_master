import torch
import torch.nn as nn
import os
from scipy import ndimage
#from Networks.classifier.get_gaussian_filter import get_gaussian_kernel
#import  math

class Classification_net(nn.Module):
    def __init__(self, in_chanel = 3,out_chanel = 43):
        super(Classification_net, self).__init__()
        self.name = os.path.realpath(__file__).split('/')[-1]
        self.conv1 = nn.Conv2d(in_chanel, 128, kernel_size=[1, 1], stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, out_chanel, kernel_size=[1, 1], stride=1, padding=0)
        self.conv_gauss = nn.Conv2d(43, 43, kernel_size=[1, 1], stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        _, idx = torch.max(x.view(x.size(0), x.size(1), -1), dim = 2)

        zero_tens=torch.zeros(2752,2500)
        idx = idx.view(2752)
        zero_tens[:, idx[:]] = 1
        zero_tens = zero_tens.view(64,43,50,50)
        #g_filter = get_gaussian_kernel(kernel_size=50, sigma=2, channels= 43)
        g_filter = torch.from_numpy(ndimage.filters.gaussian_filter(zero_tens, sigma=2)).cuda()
        with torch.no_grad():
            self.conv_gauss.weight.data = g_filter
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        return x