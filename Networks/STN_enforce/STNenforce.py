import torch
import os
import jutils

class STN(torch.nn.Module):
    def __init__(self, W_warp, H_warp, transformImage):
        super(STN, self).__init__()
        self.inDim = 3
        self.name = str(os.path.realpath(__file__).split('/')[-1])
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
        feat = self.conv2Layers(image)
        feat = torch.max( torch.max(feat, 2)[0], 2)[0].view(batchSize, -1)
        feat = self.linearLayers(feat)
        pMtrx = self.vec2mtrx(feat, warpType="affine")

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
        if warpType == "homography":
            p1, p2, p3, p4, p5, p6, p7, p8 = torch.unbind(p, dim=1)
            pMtrx = torch.stack([torch.stack([I + p1, p2, p3], dim=-1),
                                 torch.stack([p4, I + p5, p6], dim=-1),
                                 torch.stack([p7, p8, I], dim=-1)], dim=1)
        return pMtrx

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