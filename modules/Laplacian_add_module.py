import torch
import numpy as np


class Laplacian_transformation(torch.nn.Module):
    def __init__(self, level_size):
        super(Laplacian_transformation, self).__init__()
        self.pyramid_level = len(level_size)
        self.pyramid_level_size = level_size

    def forward(self, image, pMtrx, W, H):
        LP_pyr = self.Laplacian_pyr(image)
        imageWarp = self.GridCrop_batch(LP_pyr, pMtrx, W=W, H=H)
        return imageWarp

    def Laplacian_pyr(self, image):

        # Gaussian Pyramid
        GP = []
        GP.append(image)
        for i in range(self.pyramid_level - 1):
            image = torch.nn.functional.interpolate(image, size=self.pyramid_level_size[i+1], mode='bilinear', align_corners=True)
            GP.append(image)

        # Laplacian pyramid
        LP = []
        for i in range(self.pyramid_level - 1):
            UP = torch.nn.functional.interpolate(GP[i + 1], size=self.pyramid_level_size[i], mode='bilinear', align_corners=True)
            LP.append(GP[i] - UP)
        LP.append(GP[self.pyramid_level - 1])
        return LP

    def GridCrop_batch(self, LP_pyr, pMtrx, W=None, H=None):
        LP_level = []
        batch_size = pMtrx.shape[0]
        grid = torch.affine_grid_generator(pMtrx,(batch_size, 3, W, H), align_corners=True)

        # sampling with bilinear interpolation
        # x, y, h, w = original_img.size()
        # Grid Sampling
        for LP_level_img in LP_pyr:
            imageWarp = torch.nn.functional.grid_sample(LP_level_img, grid, 'bilinear')
            LP_level.append(imageWarp)

        # Collapse(Add)
        out_img = torch.zeros_like(LP_level[0])
        for LP_level_img in LP_level:
            out_img = out_img + LP_level_img

        return out_img