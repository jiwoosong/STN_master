import torch
import numpy as np


class LP_Collapsing_interp(torch.nn.Module):
    def __init__(self, pyramid_level, align_corners):
        super(LP_Collapsing_interp, self).__init__()
        self.pytramid_level = pyramid_level
        self.align_corners = align_corners

    def forward(self, image, pMtrx, W_out, H_out):
        pyr_size=[]
        size1 = tuple([16,16])
        for i in range(self.pytramid_level):
            pyr_size.append(size1)
            size1 = tuple([int(item / 2) for item in size1])

        LP_pyr,grid = self.Laplacian_pyr(image, pyr_size, pMtrx, W_out, H_out, align_corners=self.align_corners)
        imageWarp = self.Collapse_grid(LP_pyr,pyr_size)
        return imageWarp

    def Laplacian_pyr(self, image,pyr_size,pMtrx, W_out=None, H_out=None, align_corners=True):
        # Gaussian Pyramid
        batch_size = pMtrx.shape[0]
        grid = torch.affine_grid_generator(pMtrx, (batch_size, 3, W_out, H_out), align_corners=align_corners)
        image = torch.nn.functional.grid_sample(image, grid, 'bilinear', align_corners=align_corners)
        GP = []
        GP.append(image)
        for i in range(self.pytramid_level - 1):
            image = torch.nn.functional.interpolate(image, size=pyr_size[i+1], mode='bilinear', align_corners= self.align_corners)
            GP.append(image)

        # Laplacian pyramid
        LP = []
        for i in range(self.pytramid_level - 1):
            UP = torch.nn.functional.interpolate(GP[i + 1], size=pyr_size[i], mode='bilinear', align_corners= self.align_corners)
            LP.append(GP[i] - UP)
        LP.append(GP[self.pytramid_level - 1])
        return LP, grid

    def Collapse_grid(self, LP_pyr,pyr_size):
        LP_Out = LP_pyr[0]
        # sampling with bilinear interpolation
        # x, y, h, w = original_img.size()
        # Grid Sampling
        for i in range(self.pytramid_level - 1,0,-1):
            UP = torch.nn.functional.interpolate(LP_pyr[i], size=pyr_size[i-1], mode='bilinear', align_corners= self.align_corners)
            LP_Out = UP + LP_pyr[i-1]


        return LP_Out