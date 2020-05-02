import torch
import jutils
import os
from modules.Bilinear_module import bilienar_tranform

class Bilinear(torch.nn.Module):
    def __init__(self,align_corners):
        super(Bilinear, self).__init__()
        self.name = os.path.realpath(__file__).split('/')[-1]
        self.level_size = []
        self.align_corners =align_corners

    def forward(self, image, pMtrx, W, H):
        warped_image = bilienar_tranform(image, pMtrx, W=W, H=H,align_corners=self.align_corners)

        return warped_image
