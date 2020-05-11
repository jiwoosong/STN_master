import numpy as np
import os
import jutils
from torchvision.transforms import transforms
from Dataloaders.GTSRB_Loader import GTSRB
from Dataloaders.GTSRB_Loader import test_GTSRB
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def GTSRB_Loader(W, H, batch_size):
    train_root = '/home/adrenaline36/pub/db/GTSRB/data/Dataset/Train/'
    # val_root = '/home/adrenaline36/pub/db/GTSRB/data/Dataset/val/'
    test_root = '/home/adrenaline36/pub/db/GTSRB/data/Dataset/Test/'
    dataset = GTSRB(root_dir=train_root, classes=43, W=W, H=H, transform=transforms.ToTensor())
    test_dataset = test_GTSRB(root_dir=test_root, W=W, H=H, transform=transforms.ToTensor())
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = 3900
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # train/val loader, test_loader
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, validation_loader, test_loader
def GTSRB_Pert_Loader(W, H, batch_size):
    return


# def Main(geometric, classifier, optimizer, best_acc, start_iter, max_iter, W, H, batch_size):
def training(input_dic, W_in, H_in):

    train_loader, val_loader, test_loader = GTSRB_Loader(W_in, H_in,
                                                         batch_size=input_dic['batch_size'])
    it = input_dic['start_iter']
    total_epoch = int((input_dic['max_iter']-input_dic['start_iter'] + 1)*input_dic['batch_size']/len(train_loader))
    name = '_'.join(input_dic['project_path'].split('/'))
    vis = jutils.Visdom(name=name, port=input_dic['vis_port'])

    geometric = input_dic['geometric']
    classifier = input_dic['classifier']
    optim = input_dic['optim']
    criterion = input_dic['criterion']
    it_list=[]
    geometric.train()
    classifier.train()

    moving_average = 0.
    for epoch in range(total_epoch):
        for _, (img, classId, img_shape, x1y1x2y2) in enumerate(train_loader):
            if (it + 1) >= input_dic['save_iter'][-1]:
                break
            # Setting Image/Label
            it += 1
            image = img.cuda()
            label = classId.squeeze(1).cuda()

            # Forward Image
            optim.zero_grad()
            imageWarpAll = geometric(image)
            imageWarp = imageWarpAll[-1]
            output = classifier(imageWarp)

            # Backward Loss
            loss = criterion(output, label)
            moving_average += loss.item()
            loss.backward()
            optim.step()
            
            # Evaluation / Print Training circumstances
            if (it+1) % input_dic['val_iter'] == 0:
                it_list.append(it + 1)
                # update train_loss
                input_dic['train_loss_list'].append(moving_average / input_dic['val_iter'])
                # Validation & Test
                val_acc, val_mean, val_var = evaluation(geometric, classifier, val_loader, input_dic['batch_size'])
                test_acc, test_mean, test_var = evaluation(geometric, classifier, test_loader, input_dic['batch_size'])
                # update val/test error
                input_dic['val_err_list'].append((1 - val_acc) * 100)
                input_dic['test_err_list'].append((1 - test_acc) * 100)
                # vis.val_meanVar(name, val_mean, val_var, geometric.W_warp, geometric.H_warp)
                vis.test_meanVar(name, test_mean, test_var, geometric.W_warp, geometric.H_warp)
                # What if Best validation
                if (input_dic['val_best_err'] > (1 - val_acc) * 100):
                    input_dic['val_best_err'] = (1 - val_acc) * 100
                    input_dic['test_best_err'] = (1 - test_acc) * 100
                    jutils.save_best_checkpoint({
                        'iter': it_list,
                        'geometric_state_dict': geometric.state_dict(),
                        'classifier_state_dict': classifier.state_dict(),
                        'val_best_err': input_dic['val_best_err'],
                        'test_best_err': input_dic['test_best_err'],
                        'train_loss_list': input_dic['train_loss_list'],
                        'val_err_list': input_dic['val_err_list'],
                        'test_err_list': input_dic['test_err_list'],
                        'optimizer': optim.state_dict()}, save_fld=input_dic['project_path'], filename="weight")
                # if (it + 1) in input_dic['save_iter']:
                #     print_func(input_dic, it, optim, moving_average,True)
                # else:
                #     print_func(input_dic, it, optim, moving_average, False)
                moving_average = 0.

                vis.trainLoss(name, it_list, input_dic['train_loss_list'])
                vis.val_err(name, it_list, input_dic['val_err_list'])
                vis.test_err(name, it_list, input_dic['test_err_list'])

            if (it+1) in input_dic['save_iter']:
                # if (it+1) % input_dic['val_iter'] != 0:
                # if not (it + 1) % input_dic['val_iter'] == 0:
                #     print_func(input_dic, it, optim,True)
                jutils.save_checkpoint({
                    'iter': it_list,
                    'geometric_state_dict': geometric.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'val_best_err': input_dic['val_best_err'],
                    'test_best_err': input_dic['test_best_err'],
                    'train_loss_list': input_dic['train_loss_list'],
                    'val_err_list': input_dic['val_err_list'],
                    'test_err_list': input_dic['test_err_list'],
                    'optimizer': optim.state_dict()}, save_fld=input_dic['project_path'], filename="weight",num=it+1)
                print_func(input_dic, it, optim, True)

        if (it + 1) >= input_dic['save_iter'][-1]:
            print('Training Done')
            break


def evaluation(geometric, classifier, dataloader, batch_size):
    geometric.eval()
    classifier.eval()
    count = 0
    idx = []
    warped = [{}, {}]
    N = len(dataloader)
    for i, (img, classId, img_shape, x1y1x2y2) in enumerate(dataloader):
        image = img.cuda()
        label = classId.squeeze(1).cuda()

        imageWarpAll = geometric(image)
        imageWarpAll = imageWarpAll[-1]
        output = classifier(imageWarpAll)

        _, pred = output.max(dim=1)
        count += int((pred == label).sum().cpu().numpy())

        imgPert = image.detach().cpu().numpy()
        imgWarp = imageWarpAll.detach().cpu().numpy()
        
        for j in range(60):
            l = label[j].item()
            if l not in warped[0]: warped[0][l] = []
            if l not in warped[1]: warped[1][l] = []
            warped[0][l].append(imgPert[j])
            warped[1][l].append(imgWarp[j])
            idx.append(l)
    warped_cut = [{}, {}]
    x = 0
    for ind in idx[0:10]:
        if x not in warped_cut[0]: warped_cut[0][x] = []
        if x not in warped_cut[1]: warped_cut[1][x] = []
        warped_cut[0][x].append(warped[0][ind])
        warped_cut[1][x].append(warped[1][ind])
        x += 1

    warped = warped_cut
    accuracy = float(count) / (N * batch_size)
    mean = [np.array([np.mean(warped[0][l], axis=1) for l in warped[0]]),
            np.array([np.mean(warped[1][l], axis=1) for l in warped[1]])]
    var = [np.array([np.var(warped[0][l], axis=1) for l in warped[0]]),
           np.array([np.var(warped[1][l], axis=1) for l in warped[1]])]

    geometric.train()
    classifier.train()
    return accuracy, mean, var

def print_func(input_dic,it,optim, save_flag):
    if (len(input_dic['val_err_list'])>0):
        print('TRAIN[{0}/{1}] VAL+TEST[{5}/{6}]\t\t'
              '(glr:{2}clr:{3})\t'
              'loss : {4}\t'
              'VAL_err : {7}%({8}%)\t'
              'Test_err : {9}%({10}%)\t'
              'Save={11}'.format(
            input_dic['max_iter'], it,
            jutils.toGreen("{0:.7f}".format(optim.param_groups[0]['lr'])),
            jutils.toGreen("{0:.7f}".format(optim.param_groups[1]['lr'])),
            jutils.toGreen('{0:.4f}'.format(input_dic['train_loss_list'][-1])),
            int(input_dic['max_iter'] / input_dic['val_iter']), int(it / input_dic['val_iter']),
            jutils.toCyan("{0:.2f}".format(input_dic['val_err_list'][-1])),
            jutils.toRed("{0:.2f}".format(input_dic['val_best_err'])),
            jutils.toCyan("{0:.2f}".format(input_dic['test_err_list'][-1])), jutils.toRed("{0:.2f}".format(
                input_dic['test_best_err'])), str(save_flag)
        ))
    else:
        pass