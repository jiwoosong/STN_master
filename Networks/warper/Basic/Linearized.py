import torch
import os
from modules.Linearized_module import linearized_transform


class Linearized(torch.nn.Module):
    def __init__(self):
        super(Linearized, self).__init__()
        self.level_size = []
        self.name = os.path.realpath(__file__).split('/')[-1]
        # print(jutils.toCyan('-----------------------------------------'))
        # print(jutils.toCyan('Warping Module : Linearized'))

    def forward(self, image, pMtrx, W, H):
        warped_image = linearized_transform(image, pMtrx, W=self.W, H=self.H)

        return warped_image
