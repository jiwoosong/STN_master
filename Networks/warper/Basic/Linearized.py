import torch
import os
from modules.Linearized_module import linearized_transform


class Linearized(torch.nn.Module):
    def __init__(self, align_corners):
        super(Linearized, self).__init__()
        self.level_size = []
        self.name = os.path.realpath(__file__).split('/')[-1]
        self.align_corners = align_corners
        self.linearized_transform = linearized_transform(align_corners)
        # print(jutils.toCyan('-----------------------------------------'))
        # print(jutils.toCyan('Warping Module : Linearized'))

    def forward(self, image, pMtrx, W, H):
        warped_image = self.linearized_transform(image, pMtrx, W, H)

        return warped_image
