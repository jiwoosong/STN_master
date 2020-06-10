import torch
import numpy as np


class LP_After_Collapsing_interp(torch.nn.Module):
    def __init__(self, pyramid_level, align_corners):
        super(LP_After_Collapsing_interp, self).__init__()
        self.pyramid_level = pyramid_level
        self.align_corners = align_corners

    def forward(self, image, pMtrx, W_out, H_out):
        pyr_size1=[]
        pyr_size2=[]
        size1 = tuple(image.shape[-2:])
        size2 = tuple([16,16])
        for i in range(self.pyramid_level):
            pyr_size1.append(size1)
            size1 = tuple([int(item / 2) for item in size1])
            pyr_size2.append(size2)
            size2 = tuple([int(item / 2) for item in size2])

        LP_pyr = self.Laplacian_pyr(image, pyr_size1, pyr_size2,pMtrx, W_out, H_out, align_corners=self.align_corners)
        imageWarp = self.Collapse_grid(LP_pyr,pyr_size2)
        return imageWarp

    def Laplacian_pyr(self, image, pyr_size1, pyr_size2, pMtrx, W_out=None, H_out=None, align_corners=True):
        # Gaussian Pyramid
        GP = []
        GP.append(image)
        batch_size = pMtrx.shape[0]
        grid = torch.affine_grid_generator(pMtrx,(batch_size, 3, W_out, H_out), align_corners=align_corners)
        for i in range(self.pyramid_level - 1):
            image = torch.nn.functional.interpolate(image, size=pyr_size1[i+1], mode='bilinear', align_corners= self.align_corners)
            GP.append(image)

        # Laplacian pyramid
        LP = []
        GP_Grid = []
        for i in range(len(pyr_size2)):
            grid = torch.affine_grid_generator(pMtrx, (batch_size, 3, pyr_size2[i][0], pyr_size2[i][1]), align_corners=align_corners)
            imageWarp = torch.nn.functional.grid_sample(GP[i], grid, 'bilinear',align_corners=align_corners)
            GP_Grid.append(imageWarp)

        for i in range(self.pyramid_level - 1):
            UP = torch.nn.functional.interpolate(GP_Grid[i + 1], size=pyr_size2[i], mode='bilinear', align_corners= self.align_corners)
            LP.append(GP_Grid[i] - UP)
        LP.append(GP_Grid[self.pyramid_level - 1])
        return LP

    def Collapse_grid(self, LP_pyr,pyr_size):
        LP_pyr.reverse()
        pyr_size.reverse()
        out_img = LP_pyr[0]

        for i in range(self.pyramid_level-1):
            out_img = torch.nn.functional.interpolate(out_img, size=pyr_size[i+1], mode='bilinear',
                                                      align_corners=self.align_corners)
            out_img = out_img + LP_pyr[i+1]
        return out_img