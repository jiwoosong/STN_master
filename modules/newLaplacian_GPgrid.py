import torch
import numpy as np


class Laplacian_GPgrid(torch.nn.Module):
    def __init__(self, pyramid_level=3, align_corners=True):
        super(Laplacian_GPgrid, self).__init__()
        self.pyramid_level = pyramid_level
        self.align_corners = align_corners

    def forward(self, image, pMtrx, W_out, H_out):
        pyr_size = []
        size1 = tuple([W_out, H_out])
        for i in range(self.pyramid_level):
            pyr_size.append(size1)
            size1 = tuple([int(item / 2) for item in size1])

        LP_pyrWarp = self.Laplacian_gaussgridpyr(image, pMtrx, pyr_size, align_corners=self.align_corners)


        imageWarp = self.Collapse(LP_pyrWarp, pyr_size, align_corners=self.align_corners)
        return imageWarp

    def Laplacian_gaussgridpyr(self, image, pMtrx, pyr_size, align_corners):
        GP_grid = []
        batch_size = pMtrx.shape[0]

        # Gaussian Pyramid
        GP = []
        size = (batch_size,3) + pyr_size[-1]
        for i in range(self.pyramid_level - 1):
            grid = torch.affine_grid_generator(pMtrx, (batch_size, 3, pyr_size[i+1][0], pyr_size[i+1][1]), align_corners=align_corners)
            imageWarp = torch.nn.functional.grid_sample(image, grid, 'bilinear', align_corners=align_corners)
            GP.append(imageWarp)

        # GP = []
        # GP.append(image)
        # for i in range(self.pyramid_level - 1):
        #     image = torch.nn.functional.interpolate(image, size=self.pyramid_level_size[i+1], mode='bilinear', align_corners= self.align_corners)
        #     GP.append(image)

        # Laplacian pyramid
        LP = []
        for i in range(self.pyramid_level - 1):
            UP = torch.nn.functional.interpolate(GP[i + 1], size=pyr_size[i], mode='bilinear', align_corners= self.align_corners)
            LP.append(GP[i] - UP)
        LP.append(GP[self.pyramid_level - 1])
        return LP

    def Collapse(self, LP_pyr, pyr_size, align_corners=True):
        LP_pyr.reverse()
        pyr_size.reverse()
        out_img = LP_pyr[0]

        for i in range(self.pyramid_level-1):
            out_img = torch.nn.functional.interpolate(out_img, size=pyr_size[i+1], mode='bilinear',
                                                      align_corners=self.align_corners)
            out_img = out_img + LP_pyr[i+1]

        return out_img

