import torch

# def gradient_warp_module(image, pMtrx, out_shape, warper):
#     pMtrx = pMtrx / pMtrx[:, 2:3, 2:3]
#     pNew = mtrx2vec(pMtrx)
#     pMtrx = vec2mtrx(pNew, 'affine')
#     warped_image = warper(image[None], pMtrx, W = out_shape[0], H=out_shape[1], align_corners=True)
#     return warped_image

class gradient_warper():
    def __init__(self,warper):
        super(gradient_warper, self).__init__()
        self.warper = warper
    def warp_image(self, image, pMtrx, out_shape):
        pMtrx = pMtrx / pMtrx[:, 2:3, 2:3]
        pNew = self.mtrx2vec(pMtrx)
        pMtrx = self.vec2mtrx(pNew, 'affine')
        if len(image.shape)==3:
            warped_image = self.warper(image[None], pMtrx, out_shape[0], out_shape[1])
        else:
            warped_image = self.warper(image, pMtrx, out_shape[0], out_shape[1])
        return warped_image

    def vec2mtrx(self,p, warpType="affine"):
        batch_size = p.shape[0]
        if p.is_cuda:
            O = torch.zeros(batch_size, dtype=torch.float32).cuda()
            I = torch.ones(batch_size, dtype=torch.float32).cuda()
        else:
            O = torch.zeros(batch_size, dtype=torch.float32)
            I = torch.ones(batch_size, dtype=torch.float32)

        if warpType == "translation":
            tx, ty = torch.unbind(p, dim=1)
            pMtrx = torch.stack([torch.stack([I, O, tx], dim=-1),
                                 torch.stack([O, I, ty], dim=-1),
                                 torch.stack([O, O, I], dim=-1)], dim=1)
        if warpType == "similarity":
            pc, ps, tx, ty = torch.unbind(p, dim=1)
            pMtrx = torch.stack([torch.stack([I + pc, -ps, tx], dim=-1),
                                 torch.stack([ps, I + pc, ty], dim=-1),
                                 torch.stack([O, O, I], dim=-1)], dim=1)
        if warpType == "affine":
            p1, p2, p3, p4, p5, p6 = torch.unbind(p, dim=1)
            pMtrx = torch.stack([torch.stack([I + p1, p2, p3], dim=-1),
                                 torch.stack([p4, I + p5, p6], dim=-1)], dim=1)
        if warpType == "affine_3x3":
            p1, p2, p3, p4, p5, p6 = torch.unbind(p, dim=1)
            pMtrx = torch.stack([torch.stack([I + p1, p2, p3], dim=-1),
                                 torch.stack([p4, I + p5, p6], dim=-1),
                                 torch.stack([O, O, I], dim=-1)], dim=1)

        if warpType == "homography":
            p1, p2, p3, p4, p5, p6, p7, p8 = torch.unbind(p, dim=1)
            pMtrx = torch.stack([torch.stack([I + p1, p2, p3], dim=-1),
                                 torch.stack([p4, I + p5, p6], dim=-1),
                                 torch.stack([p7, p8, I], dim=-1)], dim=1)
        return pMtrx
    def compose(self,p, dp, mode):
        if mode == "affine":
            pMtrx = self.vec2mtrx(p, "affine_3x3")
            dpMtrx = self.vec2mtrx(dp, "affine_3x3")
        else:
            pMtrx = self.vec2mtrx(p, mode)
            dpMtrx = self.vec2mtrx(dp, mode)
        pMtrxNew = dpMtrx.matmul(pMtrx)
        pMtrxNew = pMtrxNew / pMtrxNew[:, 2:3, 2:3]
        pNew = self.mtrx2vec(pMtrxNew)
        return pNew
    def mtrx2vec(self,pMtrx):
        [row0, row1, row2] = torch.unbind(pMtrx, dim=1)
        [e00, e01, e02] = torch.unbind(row0, dim=1)
        [e10, e11, e12] = torch.unbind(row1, dim=1)
        [e20, e21, e22] = torch.unbind(row2, dim=1)
        p = torch.stack([e00 - 1, e01, e02, e10, e11 - 1, e12], dim=1)
        # if opt.warpType == "translation": p = torch.stack([e02, e12], dim=1)
        # if opt.warpType == "similarity": p = torch.stack([e00 - 1, e10, e02, e12], dim=1)
        # if opt.warpType == "affine": p = torch.stack([e00 - 1, e01, e02, e10, e11 - 1, e12], dim=1)
        # if opt.warpType == "homography": p = torch.stack([e00 - 1, e01, e02, e10, e11 - 1, e12, e20, e21], dim=1)
        return p


# def bilienar_tranform(image, pMtrx, W=None, H=None, align_corners=True):
#     batch_size = pMtrx.shape[0]
#     grid = torch.affine_grid_generator(pMtrx, (batch_size, 3, W, H), align_corners=align_corners)
#     # sampling with bilinear interpolation
#     imageWarp = torch.nn.functional.grid_sample(image, grid, mode="bilinear", padding_mode='zeros', align_corners=align_corners)
#     # imageWarp = linearized.grid_sample(image, grid, mode="bilinear")
#
#     return imageWarp