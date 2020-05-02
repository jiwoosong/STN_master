import torch

def bilienar_tranform(image, pMtrx, W=None, H=None, align_corners=True):
    batch_size = pMtrx.shape[0]
    grid = torch.affine_grid_generator(pMtrx, (batch_size, 3, W, H), align_corners=align_corners)
    # sampling with bilinear interpolation
    imageWarp = torch.nn.functional.grid_sample(image, grid, mode="bilinear", padding_mode='zeros', align_corners=align_corners)
    # imageWarp = linearized.grid_sample(image, grid, mode="bilinear")

    return imageWarp