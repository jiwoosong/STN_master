import torch
import modules.linearized_multisampling_release.warp.linearized as linearized


class linearized_transform(torch.nn.Module):
    def __init__(self, align_corners):
        super(linearized_transform, self).__init__()
        self.level_size = []
        self.align_corners = align_corners

    def forward(self, image, pMtrx, W, H):
        batch_size = pMtrx.shape[0]
        grid = torch.affine_grid_generator(pMtrx, (batch_size, 3, W, H), align_corners=self.align_corners)
        imageWarp = linearized.grid_sample(image, grid, mode="linearized")

        return imageWarp
#
# def linearized_transform(image, pMtrx, W=None, H=None, align_corners=True):
#     batch_size = pMtrx.shape[0]
#     grid = torch.affine_grid_generator(pMtrx, (batch_size, 3, W, H), align_corners=align_corners)
#     imageWarp = linearized.grid_sample(image, grid, mode="linearized")
#     return imageWarp