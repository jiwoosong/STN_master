import torch
import jutils
import os
import modules.newLaplacian as Laplacian_module

class Laplacian(torch.nn.Module):
    def __init__(self,align_corners, input_size, output_size, level):
        super(Laplacian, self).__init__()
        self.level_size = []
        self.name = os.path.realpath(__file__).split('/')[-1]
        self.input_size = input_size
        self.output_size = output_size

        if level is not None:
            for i in range(level):
                self.level_size.append(input_size)
                input_size = tuple([int(item / 2) for item in input_size])

        # print(jutils.toCyan('-----------------------------------------'))
        # print(jutils.toCyan('Warping Module : Laplacian'))
        # print(jutils.toCyan('LPLevel          : ' + str(self.level)))
        print(jutils.toRed('INFO : LP Level_size     : ' + str(self.level_size)))
        self.laplacian_add = Laplacian_module.newLaplacian(self.level_size,align_corners)

    def forward(self, image, pMtrx, W, H):
        warped_image = self.laplacian_add(image, pMtrx, W, H)
        return warped_image
