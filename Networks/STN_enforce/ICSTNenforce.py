import torch
import os
import jutils

class ICSTN(torch.nn.Module):
    def __init__(self, W_warp, H_warp, transformImage, level):
        super(ICSTN, self).__init__()
        self.inDim = 3
        self.name = str(os.path.realpath(__file__).split('/')[-1])
        self.level =  level
        def conv2Layer(outDim):
            conv = torch.nn.Conv2d(self.inDim, outDim, kernel_size=[7, 7], stride=1, padding=0)
            self.inDim = outDim
            return conv

        def linearLayer(outDim):
            fc = torch.nn.Linear(self.inDim, outDim)
            self.inDim = outDim
            return fc

        def maxpoolLayer(): return torch.nn.MaxPool2d([2, 2], stride=2)
        self.conv2Layers = torch.nn.Sequential(
            conv2Layer(6),torch.nn.ReLU(True),
            conv2Layer(8),torch.nn.ReLU(True),maxpoolLayer(),
            conv2Layer(32),torch.nn.ReLU(True),
            conv2Layer(1024)
        )
        self.inDim = 1024
        self.linearLayers = torch.nn.Sequential(
            linearLayer(48), torch.nn.ReLU(True),
            linearLayer(6)
        )
        self.W_warp = W_warp
        self.H_warp = H_warp
        self.transformImage = transformImage
        self.initialize(model=self, stddev=1e-01, last0=True)

    def forward(self, image):
        imageWarpAll = [image]
        batchSize = image.shape[0]
        p = torch.zeros((image.shape[0],6)).cuda()
        for l in range(4):
            pMtrx = self.vec2mtrx(p, 'affine')
            imageWarp = self.transformImage(image,pMtrx, W=self.W_warp, H=self.H_warp)
            imageWarpAll.append(imageWarp)
            feat = imageWarp
            feat = self.conv2Layers(feat)
            feat = torch.max(torch.max(feat, 2)[0], 2)[0].view(batchSize, -1)
            feat = self.linearLayers(feat)
            dp = feat
            p = self.compose(p,dp, 'affine')
        pMtrx = self.vec2mtrx(p, 'affine')
        imageWarp = self.transformImage(image, pMtrx, W=self.W_warp, H=self.H_warp)
        imageWarpAll.append(imageWarp)

        return imageWarpAll

    def vec2mtrx(self, p, warpType="affine"):
        batch_size = p.shape[0]
        O = torch.zeros(batch_size, dtype=torch.float32).cuda()
        I = torch.ones(batch_size, dtype=torch.float32).cuda()

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
    def compose(self, p, dp, mode):
        if mode == "affine":
            pMtrx = self.vec2mtrx(p, "affine_3x3")
            dpMtrx = self.vec2mtrx(dp, "affine_3x3")
        else :
            pMtrx = self.vec2mtrx(p, mode)
            dpMtrx = self.vec2mtrx(dp, mode)
        pMtrxNew = dpMtrx.matmul(pMtrx)
        pMtrxNew = pMtrxNew / pMtrxNew[:, 2:3, 2:3]
        pNew = self.mtrx2vec(pMtrxNew)
        return pNew
    def mtrx2vec(self, pMtrx):
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

    # initialize weights/biases
    def initialize(self, model, stddev, last0=False):
        print(jutils.toRed("Initialize STN : True"))
        for m in model.conv2Layers:
            if isinstance(m, torch.nn.Conv2d):
                # print("initialize_conv weights")
                m.weight.data.normal_(0, stddev)
                m.bias.data.normal_(0, stddev)
        for m in model.linearLayers:
            if isinstance(m, torch.nn.Linear):
                if last0 and m is model.linearLayers[-1]:
                    # print("initialize_last_fc weights")
                    m.weight.data.zero_()
                    m.bias.data.zero_()
                else:
                    # print("initialize_fc weights")
                    m.weight.data.normal_(0, stddev)
                    m.bias.data.normal_(0, stddev)