import torch
import numpy as np


class newLaplacian(torch.nn.Module):
    def __init__(self, pyramid_level, align_corners):
        super(newLaplacian, self).__init__()
        self.pytramid_level = pyramid_level
        self.align_corners = align_corners

    def forward(self, image, pMtrx, W_out, H_out):
        pyr_size=[]
        size1 = tuple(image.shape[-2:])
        for i in range(self.pytramid_level):
            pyr_size.append(size1)
            size1 = tuple([int(item / 2) for item in size1])

        LP_pyr = self.Laplacian_pyr(image, pyr_size)
        imageWarp = self.Collapse_grid(LP_pyr, pMtrx, W_out, H_out, align_corners=self.align_corners)
        return imageWarp

    def Laplacian_pyr(self, image,pyr_size):
        # Gaussian Pyramid
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
        return LP

    def Collapse_grid(self, LP_pyr, pMtrx, W_out=None, H_out=None, align_corners=True):
        LP_level = []
        batch_size = pMtrx.shape[0]
        grid = torch.affine_grid_generator(pMtrx,(batch_size, 3, W_out, H_out), align_corners=align_corners)

        # sampling with bilinear interpolation
        # x, y, h, w = original_img.size()
        # Grid Sampling
        for LP_level_img in LP_pyr:
            imageWarp = torch.nn.functional.grid_sample(LP_level_img, grid, 'bilinear',align_corners=align_corners)
            LP_level.append(imageWarp)

        # Collapse(Add)
        out_img = torch.zeros_like(LP_level[0])
        for LP_level_img in LP_level:
            out_img = out_img + LP_level_img

        return out_img