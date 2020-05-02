import torch
import termcolor
import numpy as np
import visdom
import torch.nn.functional as F
import os

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
                             torch.stack([p4, I + p5, p6], dim=-1),
                             torch.stack([O, O, I], dim=-1)], dim=1)
    if warpType == "homography":
        p1, p2, p3, p4, p5, p6, p7, p8 = torch.unbind(p, dim=1)
        pMtrx = torch.stack([torch.stack([I + p1, p2, p3], dim=-1),
                             torch.stack([p4, I + p5, p6], dim=-1),
                             torch.stack([p7, p8, I], dim=-1)], dim=1)

    if warpType == "affine_torch":
        p1, p2, p3, p4, p5, p6 = torch.unbind(p, dim=1)
        pMtrx = torch.stack([torch.stack([I + p1, p2, p3], dim=-1),
                             torch.stack([p4, I + p5, p6], dim=-1)])
    return pMtrx

import PIL.Image as Image
import matplotlib.cm as mpl_color_map
import copy


def load_state(net, param):
    model_dict = net.state_dict()
    for name, param in param.items():
        if name not in model_dict:
            continue
        else:
            model_dict[name].copy_(param)
    return net
def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image



# convert to colored strings
def toRed(content): return termcolor.colored(content,"red",attrs=["bold"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["bold"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["bold"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["bold"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["bold"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold"])


class Visdom():
    def __init__(self,name, port=8097):
        self.vis = visdom.Visdom(port=port,use_incoming_socket=False, env=name)
        self.trainLossInit = True
        self.testLossInit = True
        self.eval_flag = True
        self.legends = []
        self.meanVarInit = True
    def tileImages(self,images,H,W,HN,WN):
        assert(len(images)==HN*WN)
        images = images.reshape([HN,WN,-1,H,W])
        images = [list(i) for i in images]
        imageBlocks = np.concatenate([np.concatenate(row,axis=2) for row in images],axis=1)
        return imageBlocks
    def trainLoss(self,name,it,loss):
        # loss = float(loss.detach().cpu().numpy())
        if self.trainLossInit:
            self.vis.line(Y=np.array(loss),X=np.array(it),win="{0}_trainloss".format(name),
                          opts={ "title": "(TRAIN_loss)"})
            # self.trainLossInit = False
        else: self.vis.line(Y=np.array(loss),X=np.array(it),win=name+"_trainloss")
    def val_err(self,name,it,loss):
        if self.testLossInit:
            self.vis.line(Y=np.array(loss),X=np.array(it),win="{0}_val_err".format(name),
                          opts={ "title": "(VAL_error)" })
            # self.testLossInit = False
        else: self.vis.line(Y=np.array(loss),X=np.array(it),win=name+"_test_err")
    def test_err(self,name,it,loss):
        if self.testLossInit:
            self.vis.line(Y=np.array(loss),X=np.array(it),win="{0}_testloss".format(name),
                          opts={ "title": "(TEST_error)" })
            # self.testLossInit = False
        else: self.vis.line(Y=np.array(loss),X=np.array(it),win=name+"_testloss")
    def test_meanVar(self,name,mean,var,W, H):
        mean = [self.tileImages(m,H,W,1,10) for m in mean]
        var = [self.tileImages(v,H,W,1,10)*3 for v in var]
        self.vis.image(mean[0].clip(0,1),win="{0}_meaninit".format(name), opts={ "title": "(mean_init)"})
        self.vis.image(mean[1].clip(0,1),win="{0}_meanwarped".format(name), opts={ "title": "(TEST_mean_warped)"})
        self.vis.image(var[0].clip(0,1),win="{0}_varinit".format(name), opts={ "title": "(var_init)"})
        self.vis.image(var[1].clip(0,1),win="{0}_varwarped".format(name), opts={ "title": "(TEST_var_warped)"})
    def val_meanVar(self,name,mean,var,W, H):
        mean = [self.tileImages(m,H,W,1,10) for m in mean]
        var = [self.tileImages(v,H,W,1,10)*3 for v in var]
        self.vis.image(mean[0].clip(0,1),win="{0}_meaninit".format(name), opts={ "title": "(mean_init)"})
        self.vis.image(mean[1].clip(0,1),win="{0}_meanwarped".format(name), opts={ "title": "(VAL_mean_warped)"})
        self.vis.image(var[0].clip(0,1),win="{0}_varinit".format(name), opts={ "title": "(var_init)"})
        self.vis.image(var[1].clip(0,1),win="{0}_varwarped".format(name), opts={ "title": "(VAL_var_warped)"})

    def Eval_plot(self,name,title,it,loss, win):
        if self.eval_flag:
            self.vis.close()
            self.legends.append("{0}_testloss".format(name))
            win = self.vis.line(Y=np.array(loss),X=np.array(it), win = 'Evaluation' , name="{0}_testloss".format(name),
                          opts={"title": title + "(TEST_error)",'legend':self.legends})
            self.eval_flag = False
        else:
            self.legends.append("{0}_testloss".format(name))
            win = self.vis.line(Y=np.array(loss), X=np.array(it), win=win, name="{0}_testloss".format(name),
                                update='append', opts={"title": title + "(TEST_error)", 'legend': self.legends})
        return win

def vis_image4(vis,tensor_image , win=None, title=""):
    # [B,C,H,W] tensor -> B*[C,H,W] numpy
    img = F.interpolate(tensor_image.unsqueeze(0), size=(200, 200),mode='bilinear')
    img = img.detach().cpu().numpy()
    # batch_size = img.shape[0]
    id = vis.images(img,opts=dict(title=title),win=win)
    return id

def save_checkpoint(state, save_fld, filename, num):
    save_path = os.path.join(save_fld,'weight')
    if not os.path.exists(save_path):
        print(toRed('Create weight save Folder...'))
        os.makedirs(save_path)
    torch.save(state, save_path +'/'+ filename +'_' + str(num)+'.pth')


def save_best_checkpoint(state, save_fld, filename):
    save_path = os.path.join(save_fld, 'weight')
    if not os.path.exists(save_path):
        print(toRed('Create weight save Folder...'))
        os.makedirs(save_path)
    torch.save(state, save_path + '/best_' + filename+'.pth')


def initial_checkpoint_size(state, save_fld):
    save_path = os.path.join(save_fld, 'weight')
    if not os.path.exists(save_path):
        print(toRed('Create weight save Folder...'))
        os.makedirs(save_path)
    torch.save(state, save_path + '/temp.pth')
    data_size = os.path.getsize(save_path+"/temp.pth")/1e6
    os.remove(save_path + '/temp.pth')
    return data_size


# def save_checkpoint(state, is_best, filename):
#     if not os.path.exists('./save/'):
#         os.mkdir('save')
#     torch.save(state, 'save/'+filename)
#     if is_best:
#         torch.save(state, 'save/'+'best_' + filename)

# def load_state(net, param):
#     model_dict = net.state_dict()
#     for name, param in param.items():
#         if name not in model_dict:
#             continue
#         #if name == 'conv1.weight':
#         #    continue
#         else:
#             model_dict[name].copy_(param)

# def vec2mtrx(p, warpType="affine"):
#     batch_size = p.shape[0]
#     O = torch.zeros(batch_size, dtype=torch.float32).cuda()
#     I = torch.ones(batch_size, dtype=torch.float32).cuda()
#
#     if warpType == "translation":
#         tx, ty = torch.unbind(p, dim=1)
#         pMtrx = torch.stack([torch.stack([I, O, tx], dim=-1),
#                              torch.stack([O, I, ty], dim=-1),
#                              torch.stack([O, O, I], dim=-1)], dim=1)
#     if warpType == "similarity":
#         pc, ps, tx, ty = torch.unbind(p, dim=1)
#         pMtrx = torch.stack([torch.stack([I + pc, -ps, tx], dim=-1),
#                              torch.stack([ps, I + pc, ty], dim=-1),
#                              torch.stack([O, O, I], dim=-1)], dim=1)
#     if warpType == "affine":
#         p1, p2, p3, p4, p5, p6 = torch.unbind(p, dim=1)
#         pMtrx = torch.stack([torch.stack([I + p1, p2, p3], dim=-1),
#                              torch.stack([p4, I + p5, p6], dim=-1),
#                              torch.stack([O, O, I], dim=-1)], dim=1)
#     if warpType == "homography":
#         p1, p2, p3, p4, p5, p6, p7, p8 = torch.unbind(p, dim=1)
#         pMtrx = torch.stack([torch.stack([I + p1, p2, p3], dim=-1),
#                              torch.stack([p4, I + p5, p6], dim=-1),
#                              torch.stack([p7, p8, I], dim=-1)], dim=1)
#     return pMtrx

# def transformImage(image, pMtrx, W=28, H=28):
#     batch_size = pMtrx.shape[0]
#
#     refMtrx = torch.from_numpy(np.eye(3).astype(np.float32)).cuda()
#     refMtrx = refMtrx.repeat(batch_size, 1, 1)
#     transMtrx = refMtrx.matmul(pMtrx)
#     # warp the canonical coordinates
#     X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
#     X, Y = X.flatten(), Y.flatten()
#     XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T
#     XYhom = np.tile(XYhom, [batch_size, 1, 1]).astype(np.float32)
#     XYhom = torch.from_numpy(XYhom).cuda()
#     XYwarpHom = transMtrx.matmul(XYhom)
#     XwarpHom, YwarpHom, ZwarpHom = torch.unbind(XYwarpHom, dim=1)
#     Xwarp = (XwarpHom / (ZwarpHom + 1e-8)).reshape(batch_size, H, W)
#     Ywarp = (YwarpHom / (ZwarpHom + 1e-8)).reshape(batch_size, H, W)
#     grid = torch.stack([Xwarp, Ywarp], dim=-1)
#     # sampling with bilinear interpolation
#     imageWarp = torch.nn.functional.grid_sample(image, grid, mode="bilinear")
#     return imageWarp


# def vis_image4(vis,tensor_image , win=None, title=""):
#     # [B,C,H,W] tensor -> B*[C,H,W] numpy
#     img = F.interpolate(tensor_image.unsqueeze(0), size=(200, 200),mode='bilinear')
#
#     img = img.detach().cpu().numpy()
#     # batch_size = img.shape[0]
#     id = vis.images(img,opts=dict(title=title),win=win)
#     return id