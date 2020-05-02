import torch
import jutils
import os
import modules.Laplacian_add_module as Laplacian_add

class Laplacian(torch.nn.Module):
    def __init__(self):
        super(Laplacian, self).__init__()
        self.level_size = []
        self.name = os.path.realpath(__file__).split('/')[-1]
        level=4
        if level is not None:
            for i in range(level):
                self.level_size.append((W, H))
                W = int(W / 2)
                H = int(H / 2)
        # print(jutils.toCyan('-----------------------------------------'))
        # print(jutils.toCyan('Warping Module : Laplacian'))
        # print(jutils.toCyan('Level          : ' + str(self.level)))
        # print(jutils.toRed('Level_size     : ' + str(self.level_size)))
        self.laplacian_add = Laplacian_add.Laplacian_transformation(self.level_size)

    def forward(self, image, pMtrx, W, H):
        warped_image = self.laplacian_add(image, pMtrx, W, H)
        return warped_image
