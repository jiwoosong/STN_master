import torch
import jutils
import os
from modules.Bilinear_module import bilienar_tranform

class Bilinear(torch.nn.Module):
    def __init__(self,align_corners):
        super(Bilinear, self).__init__()
        self.name = os.path.realpath(__file__).split('/')[-1]
        self.level_size = []
        self.bilienar_tranform =bilienar_tranform(align_corners)

    def forward(self, image, pMtrx, W, H):
        warped_image = self.bilienar_tranform(image, pMtrx, W, H)

        return warped_image
