import torch
import numpy as np


class LP_Multiscale_Collapsing_GS(torch.nn.Module):
    def __init__(self, pyramid_level, align_corners):
        super(LP_Multiscale_Collapsing_GS, self).__init__()
        self.pytramid_level = pyramid_level
        self.align_corners = align_corners

    def forward(self, image, pMtrx, W_out, H_out):
        pyr_size=[]
        size1 = tuple([16,16])
        for i in range(self.pytramid_level):
            pyr_size.append(size1)
            size1 = tuple([int(item / 2) for item in size1])

        LP_pyr,grid = self.Laplacian_pyr(image, pyr_size ,pMtrx,align_corners=self.align_corners)
        imageWarp = self.Collapse_grid(LP_pyr,grid)
        return imageWarp

    def Laplacian_pyr(self, image,pyr_size, pMtrx, align_corners=True):
        # Gaussian Pyramid
        batch_size = pMtrx.shape[0]
        grid_image = []
        grid_list = []
        for i in range(len(pyr_size)):
            grid = torch.affine_grid_generator(pMtrx, (batch_size, 3, pyr_size[i][0], pyr_size[i][1]), align_corners=align_corners)
            image = torch.nn.functional.grid_sample(image, grid, 'bilinear', align_corners=align_corners)
            grid_list.append(grid)
            grid_image.append(image)
        GP = []
        GP.append(grid_image[0])
        for i in range(self.pytramid_level - 1):
            image = torch.nn.functional.interpolate(grid_image[i], size=pyr_size[i+1], mode='bilinear', align_corners= self.align_corners)
            GP.append(image)

        # Laplacian pyramid
        LP = []
        for i in range(self.pytramid_level - 1):
            UP = torch.nn.functional.interpolate(GP[i + 1], size=pyr_size[i], mode='bilinear', align_corners= self.align_corners)
            LP.append(GP[i] - UP)
        LP.append(GP[self.pytramid_level - 1])
        return LP, grid_list

    def Collapse_grid(self, LP_pyr, grid, align_corners=True):
        LP_level = []
        # sampling with bilinear interpolation
        # x, y, h, w = original_img.size()
        # Grid Sampling
        for LP_level_img in LP_pyr:
            imageWarp = torch.nn.functional.grid_sample(LP_level_img, grid[0], 'bilinear',align_corners=align_corners)
            LP_level.append(imageWarp)

        out_img = torch.zeros_like(LP_pyr[0])
        for LP_level_img in LP_level:
            out_img = out_img + LP_level_img

        return out_img