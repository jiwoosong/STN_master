import torch
import numpy as np


class Laplacian_transformation_downadd(torch.nn.Module):
    def __init__(self, level_size,align_corners):
        super(Laplacian_transformation_downadd, self).__init__()
        self.pyramid_level = len(level_size)
        self.pyramid_level_size = level_size
        self.align_corners = align_corners

    def forward(self, image, pMtrx, W, H):
        LP_pyr = self.Laplacian_pyr(image)
        imageWarp = self.GridCrop_batch_add(LP_pyr, pMtrx, W=W, H=H, align_corners=self.align_corners)
        return imageWarp

    def Laplacian_pyr(self, image):

        # Gaussian Pyramid
        GP = []
        GP.append(image)
        for i in range(self.pyramid_level - 1):
            image = torch.nn.functional.interpolate(image, size=self.pyramid_level_size[i+1], mode='bilinear', align_corners= self.align_corners)
            GP.append(image)

        # Laplacian pyramid
        LP = []
        for i in range(self.pyramid_level - 1):
            UP = torch.nn.functional.interpolate(GP[i + 1], size=self.pyramid_level_size[i], mode='bilinear', align_corners= self.align_corners)
            LP.append(GP[i] - UP)
        LP.append(GP[self.pyramid_level - 1])
        return LP

    def GridCrop_batch_add(self, LP_pyr, pMtrx, W=None, H=None, align_corners=True):
        LP_level = []
        batch_size = pMtrx.shape[0]
        grid = torch.affine_grid_generator(pMtrx,(batch_size, 3, W, H), align_corners=align_corners)

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


class Laplacian_transformation_down1by1(torch.nn.Module):
    def __init__(self, level_size,align_corners):
        super(Laplacian_transformation_down1by1, self).__init__()
        self.pyramid_level = len(level_size)
        self.conv1by1 = torch.nn.Conv2d(self.pyramid_level*3, 3, kernel_size=[1, 1], stride=1, padding=0)
        self.pyramid_level_size = level_size
        self.align_corners = align_corners

    def forward(self, image, pMtrx, W, H):
        LP_pyr = self.Laplacian_pyr(image)
        imageWarp = self.GridCrop_batch_1by1(LP_pyr, pMtrx, W=W, H=H, align_corners=self.align_corners)
        return imageWarp

    def Laplacian_pyr(self, image):

        # Gaussian Pyramid
        GP = []
        GP.append(image)
        for i in range(self.pyramid_level - 1):
            image = torch.nn.functional.interpolate(image, size=self.pyramid_level_size[i+1], mode='bilinear', align_corners= self.align_corners)
            GP.append(image)

        # Laplacian pyramid
        LP = []
        for i in range(self.pyramid_level - 1):
            UP = torch.nn.functional.interpolate(GP[i + 1], size=self.pyramid_level_size[i], mode='bilinear', align_corners= self.align_corners)
            LP.append(GP[i] - UP)
        LP.append(GP[self.pyramid_level - 1])
        return LP

    def GridCrop_batch_1by1(self, LP_pyr, pMtrx, W=None, H=None, align_corners=True):
        LP_level = []
        batch_size = pMtrx.shape[0]
        grid = torch.affine_grid_generator(pMtrx,(batch_size, 3, W, H), align_corners=align_corners)

        # sampling with bilinear interpolation
        # x, y, h, w = original_img.size()
        # Grid Sampling
        for LP_level_img in LP_pyr:
            imageWarp = torch.nn.functional.grid_sample(LP_level_img, grid, 'bilinear', align_corners=align_corners)
            LP_level.append(imageWarp)
        LP_level = torch.cat(LP_level, dim=1)
        # Collapse(Add)
        out_img = self.conv1by1(LP_level)

        return out_img





class Laplacian_transformation_grid_add(torch.nn.Module):
    def __init__(self, level_size,align_corners):
        super(Laplacian_transformation_grid_add, self).__init__()
        self.pyramid_level = len(level_size)
        self.pyramid_level_size = level_size
        self.align_corners = align_corners

    def forward(self, image, pMtrx, W, H):
        LP_pyr = self.Laplacian_pyr(image)
        imageWarp = self.GridCrop_batch(LP_pyr, pMtrx, W=W, H=H)
        return imageWarp

    def Laplacian_pyr(self, image):

        # Gaussian Pyramid
        GP = []
        GP.append(image)
        for i in range(self.pyramid_level - 1):
            image = torch.nn.functional.interpolate(image, size=self.pyramid_level_size[i+1], mode='bilinear', align_corners=self.align_corners)
            GP.append(image)

        # Laplacian pyramid
        LP = []
        for i in range(self.pyramid_level - 1):
            UP = torch.nn.functional.interpolate(GP[i + 1], size=self.pyramid_level_size[i], mode='bilinear', align_corners=self.align_corners)
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