import torch
import torch.nn as nn
import os
from scipy import ndimage
# from Networks.classifier.get_gaussian_filter import get_gaussian_kernel
import  math

class Classification_net(nn.Module):
    def __init__(self, in_chanel = 3,out_chanel = 43):
        super(Classification_net, self).__init__()
        self.name = os.path.realpath(__file__).split('/')[-1]
        self.conv1 = nn.Conv2d(in_chanel, 128, kernel_size=[1, 1], stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, out_chanel, kernel_size=[1, 1], stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        _, idx = torch.max(x.view(x.size(0), x.size(1), -1), dim = 2)

        #zero_tens = torch.zeros((64,43,50,50))
        zero_tens=torch.zeros(2752,2500)
        #zero_tens = zero_tens.view(2752,2500)
        idx = idx.view(2752)
        zero_tens[:, idx[:]] = 1
        zero_tens = zero_tens.view(64,43,50,50)
        #g_filter = get_gaussian_kernel(zero_tens, kernel_size=50, sigma=2, channels= 43)
        g_filter = torch.from_numpy(ndimage.filters.gaussian_filter(zero_tens, sigma=2)).cuda()
        x = torch.mul(x, g_filter)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        return x

    def get_gaussian_kernel(self, zero_grid, kernel_size=50, sigma=2, channels=3):
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((zero_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter
