import os
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

from modules.linearized_multisampling_release.gradient_visualizer import gradient_visualizer
from modules.linearized_multisampling_release.warp import sampling_helper
from modules.linearized_multisampling_release.utils import utils
import matplotlib
from modules.gradient_warp_module import gradient_warper
matplotlib.use('TkAgg')

image_path = "modules/linearized_multisampling_release/notebook_data/cute.jpg"
cute_cat = imageio.imread(image_path)
cute_cat = cute_cat/255.0
# plt.imshow(cute_cat)
cute_cat = utils.np_img_to_torch_img(cute_cat).float()
# plt.show()

trans_mat = torch.tensor([[[0.6705, 0.4691, -0.1369], [-0.4691,0.6705, -0.0432],[0.0000,0.0000,1.0000]]], device='cpu')
out_shape = [128,128]
#
# bilinear_sampler = sampling_helper.DifferentiableImageSampler('bilinear','zeros')
# bilinear_transformed_image = bilinear_sampler.warp_image(cute_cat,trans_mat,out_shape)
# bilinear_transformed_image = utils.torch_img_to_np_img(bilinear_transformed_image)
# # plt.imshow(bilinear_transformed_image[0])
# # plt.show()
#
# linearized_sampler = sampling_helper.DifferentiableImageSampler('linearized','zeros')
# linearized_transformed_image = linearized_sampler.warp_image(cute_cat,trans_mat,out_shape)
# linearized_transformed_image = utils.torch_img_to_np_img(linearized_transformed_image)
# # plt.imshow(linearized_transformed_image[0])
# # plt.show()


# from modules.Bilinear_module import bilienar_tranform
# ours_bilinear_sampler = gradient_warper(bilienar_tranform(align_corners=True))
# ours_bilinear_transformed_image = ours_bilinear_sampler.warp_image(cute_cat,trans_mat,out_shape)
# ours_bilinear_transformed_image = utils.torch_img_to_np_img(ours_bilinear_transformed_image)
# # plt.imshow(ours_bilinear_transformed_image[0])
# # plt.show()


# from modules.Linearized_module import linearized_transform
# ours_linearized_sampler = gradient_warper(linearized_transform(align_corners=True))
# ours_linearized_transformed_image = ours_linearized_sampler.warp_image(cute_cat,trans_mat,out_shape)
# ours_linearized_transformed_image = utils.torch_img_to_np_img(ours_linearized_transformed_image)



#from modules.Laplacian_add_module import Laplacian_transformation_downadd
#from modules.newLaplacian import newLaplacian
#from modules.Laplacian_Collasing_GS import Laplacian_one
#from modules.Laplacian_two import Laplacian_two
#from modules.Laplacian_thr import Laplacian_thr
#from modules.Laplacian_four import Laplacian_four
from modules.LP_Collapsing_GS import LP_Collapsing_GS
#from modules.newLaplacian_GPgrid import Laplacian_GPgrid
# level=3
# level_size=[]
# W_in = cute_cat.shape[-2]
# H_in = cute_cat.shape[-1]
# if level is not None:
#     for i in range(level):
#         level_size.append((W_in,H_in))
#         W_in = int(W_in / 2)
#         H_in = int(H_in / 2)
# ours_laplacian_add_sampler = gradient_warper(Laplacian_transformation_downadd(level_size=level_size, align_corners=False))
# ours_laplacian_add_transformed_image = ours_laplacian_add_sampler.warp_image(cute_cat,trans_mat,out_shape)
# ours_laplacian_add_transformed_image = utils.torch_img_to_np_img(ours_laplacian_add_transformed_image)

class FakeOptions():
    pass
opt = FakeOptions()
opt.padding_mode = 'zeros'
opt.grid_size = 10
opt.optim_criterion = 'mse'
opt.optim_lr = 1e-2
opt.out_shape = [16,16]

# # gradient_visualizer_instance = gradient_visualizer.GradientVisualizer(opt)
# # gradient_visualizer_instance.draw_gradient_grid(cute_cat[None], bilinear_sampler)
# # pass
#
# gradient_visualizer_instance = gradient_visualizer.GradientVisualizer(opt)
# gradient_visualizer_instance.draw_gradient_grid(cute_cat[None], ours_bilinear_sampler)
# pass
#
# # gradient_visualizer_instance = gradient_visualizer.GradientVisualizer(opt)
# # gradient_visualizer_instance.draw_gradient_grid(cute_cat[None], linearized_sampler)
# # pass
#
# gradient_visualizer_instance = gradient_visualizer.GradientVisualizer(opt)
# gradient_visualizer_instance.draw_gradient_grid(cute_cat[None], ours_linearized_sampler)

# pass
#
#
# gradient_visualizer_instance = gradient_visualizer.GradientVisualizer(opt)
# gradient_visualizer_instance.draw_gradient_grid(cute_cat[None], ours_laplacian_add_sampler)
# pass

for level in range(1,6):

    level_size = []
    W_in = cute_cat.shape[-2]
    H_in = cute_cat.shape[-1]
    #W_in = 19
    #H_in = 19
    if level is not None:
        for i in range(level):
            level_size.append((W_in, H_in))
            W_in = int(W_in / 2)
            H_in = int(H_in / 2)
    ours_laplacian_add_sampler = gradient_warper(LP_Collapsing_GS(pyramid_level=level, align_corners=True))
    #ours_laplacian_add_sampler = gradient_warper(newLaplacian(pyramid_level=level, align_corners=True))
    #ours_laplacian_add_sampler = gradient_warper(Laplacian_GPgrid(pyramid_level=level, align_corners=True))
    #ours_laplacian_add_sampler = gradient_warper(Laplacian_two(pyramid_level=level, align_corners=True))
    #ours_laplacian_add_sampler = gradient_warper(Laplacian_thr(pyramid_level=level, align_corners=True))
    #ours_laplacian_add_sampler = gradient_warper(Laplacian_four(pyramid_level=level, align_corners=True))
    # ours_laplacian_add_transformed_image = ours_laplacian_add_sampler.warp_image(cute_cat, trans_mat, out_shape)
    # ours_laplacian_add_transformed_image = utils.torch_img_to_np_img(ours_laplacian_add_transformed_image)
    gradient_visualizer_instance = gradient_visualizer.GradientVisualizer(opt)
    gradient_visualizer_instance.draw_gradient_grid(cute_cat[None], ours_laplacian_add_sampler)
    a=1
    #gradient_visualizer_instance = gradient_visualizer.GradientVisualizer(opt)
    #gradient_visualizer_instance.draw_gradient_grid(cute_cat[None], ours_laplacian_add_sampler)
