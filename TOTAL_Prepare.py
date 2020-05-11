import torch
import os
import glob
import re
import jutils
# from modules.Interpolation_Module.ICSTN_module import util,data,graph,warp,util,options
import matplotlib.pyplot as plt
import matplotlib
import shutil
matplotlib.use('TkAgg')

def Prepare_Training2(input_dic):
    '''
    Testing Name : 2020_03_22_STN~~
    unet_name : Half_maxpool_add_3ch_3lv~~
    Tuning_name : Adam_0.001~~
    '''
    print('Test_name                    : ' + jutils.toGreen(input_dic['test_name']))
    ''' STN_Type / Warping_Type / Tuning_Type'''
    geometric = input_dic['geometric'].cuda()
    classifier = input_dic['classifier'].cuda()

    STN_Type = geometric.name.split('.')[0]
    warp_class_Type = geometric.transformImage.name.split('.')[0] + '_' + classifier.name.split('.')[0] + input_dic['message']
    Tuning_Type = input_dic['optimizer_name'] + '_g' + '%.0e'%(input_dic['glr']) + '_c' + '%.0e'%(input_dic['clr']) + '_wd' +  '%.0e'%(input_dic['wd'])
    print('# STN_Type                   : ' + jutils.toGreen(STN_Type))
    print('# warp_class_Type            : ' + jutils.toRed(warp_class_Type))


    ''' Setting Optimizer '''
    optimList = [{"params": geometric.parameters(), "lr": input_dic['glr']},
                 {"params": classifier.parameters(), "lr": input_dic['clr']}]
    optim = Setting_Optimizer(input_dic['optimizer_name'],input_dic['wd'], optimList)
    print('Tuning_Type                  : '+ jutils.toRed(Tuning_Type))
    print('Load                         : ' + str(input_dic['load_flag']))




    '''Save Folder'''
    project_path = os.path.join(input_dic['test_name'], STN_Type, warp_class_Type, Tuning_Type)
    if not os.path.isdir(project_path):
        os.makedirs(os.path.join(project_path))
        print(jutils.toMagenta('###Create New Project...###'))
        # print(jutils.toMagenta('Project : ' + jutils.toGreen(test_name) +'/' + jutils.toGreen(STN_Type) + '/' + jutils.toRed(Warping_Type) + '/' + jutils.toRed(Tuning_Type)))
        val_best_err = 100.
        test_best_err =100.
        start_iter = 0
        it=[]
        train_loss_list = []
        val_err_list = []
        test_err_list = []
    else:
        print(jutils.toMagenta("###Project Already Exists....###"))
        if input_dic['load_flag']:
            print(jutils.toRed('###Load Project...###'))
            max_val = get_last_weight(project_path)
            if max_val is not None:
                net_params = torch.load(os.path.join(project_path, 'weight', 'weight_'+str(max_val))+'.pth',map_location='cpu')

                geometric = jutils.load_state(geometric, net_params['geometric_state_dict']).cuda()
                classifier = jutils.load_state(classifier, net_params['classifier_state_dict']).cuda()

                val_best_err = net_params['val_best_err']
                test_best_err = net_params['test_best_err']
                train_loss_list = net_params['train_loss_list']
                val_err_list = net_params['val_err_list']
                test_err_list = net_params['test_err_list']
                it = net_params['iter']
                start_iter = it[-1]
                optim.load_state_dict(net_params['optimizer'])
                for state in optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

                print(jutils.toMagenta("Loading Suceed ..."))
            else:
                print(jutils.toMagenta("Loading Failed ..."))
                raise
        else:
            print(jutils.toRed('###Delete and Create Project...###'))
            shutil.rmtree(project_path)
            os.makedirs(os.path.join(project_path))
            it = []
            val_best_err = 100.
            test_best_err = 100.
            start_iter = 0
            train_loss_list = []
            val_err_list = []
            test_err_list = []
        # print('Project : ' + STN_Type + '/' + jutils.toRed(Warping_Type) + '/' + jutils.toGreen(Tuning_Type))

    ''' Printing Exist Project & Last Iterations '''
    Print_Test_name_lists(input_dic['test_name'],STN_Type,warp_class_Type,Tuning_Type)

    output_dic = input_dic
    # Updates Network
    output_dic['geometric'] = geometric
    output_dic['classifier'] = classifier
    output_dic['optim'] = optim
    # Update Start / Max iterations / batchsize
    output_dic['iter'] = it
    output_dic['start_iter'] = start_iter
    output_dic['batch_size'] = input_dic['batch_size']
    # Update visdom_name / losses / Errors
    output_dic['project_path'] = project_path #'_'.join(project_path.split('/'))
    output_dic['train_loss_list'] = train_loss_list
    output_dic['val_err_list'] = val_err_list
    output_dic['test_err_list'] = test_err_list
    output_dic['val_best_err'] = val_best_err
    output_dic['test_best_err'] = test_best_err

    save_iter = []
    prev = 0
    for it, period in output_dic['save_iter']:
        save_iter = save_iter + list(range(prev,it,period))
        prev = it

    output_dic['save_iter'] = save_iter


    data_size = jutils.initial_checkpoint_size(geometric.state_dict(),project_path)
    data_size += jutils.initial_checkpoint_size(classifier.state_dict(), project_path)
    data_size += jutils.initial_checkpoint_size(optim.state_dict(), project_path)
    print('Total save size      : '+ jutils.toRed(str(data_size*len(save_iter)) + ' MB'))
    return output_dic


def get_last_weight(project_path):
    load_path = os.path.join(project_path, 'weight')
    file_list = glob.glob(load_path + '/weight*')
    file_list = [int(re.findall('\d+', file.split('/')[-1])[0]) for file in file_list]
    if len(file_list)>0:
        max_val = max(file_list)
        return max_val
    else:
        return None

def Print_Test_name_lists(test_name,STN_Type,warp_class_Type,Tuning_Type):
    it=0
    if os.path.isdir(os.path.join(test_name)):
        print('----------------Searching----------------')
        for STN_Type_names in os.listdir(os.path.join(test_name)):
            # print('STN_types : ' + STN_Type_names)
            for warp_class_type_names in os.listdir(os.path.join(test_name, STN_Type_names)):
                # print(' ∟Project : ' + Warping_type_names)
                for Tuning_Type_names in os.listdir(os.path.join(test_name, STN_Type_names, warp_class_type_names)):
                    it += 1
                    if(STN_Type==STN_Type_names and warp_class_Type==warp_class_type_names and Tuning_Type == Tuning_Type_names):
                        print(' >> '+ jutils.toGreen(test_name)+ '/' + jutils.toGreen(STN_Type_names) + '/' + jutils.toRed(warp_class_type_names) + '/' + jutils.toRed(
                            Tuning_Type_names) + "      Last_Save : " + str(
                            get_last_weight(
                                os.path.join(test_name, STN_Type_names, warp_class_type_names, Tuning_Type_names))))
                    else:
                        print('    '+ test_name+ '/' + STN_Type_names + '/' + warp_class_type_names + '/' +
                            Tuning_Type_names + "      Last_Save : " + str(
                            get_last_weight(
                                os.path.join(test_name, STN_Type_names, warp_class_type_names, Tuning_Type_names))))
        print('-----------------------------------------')
    else:
        print(
            jutils.toRed('ERR : No Test_name [' + test_name + '] (ex. STN/ICSTN/STN_Perturbation/ICSTN_Perturbation)'))
        raise

def Setting_Optimizer(optimizer_name,w_d, optimList):
    if optimizer_name == 'Adam':
        if w_d is not None:
            optim = torch.optim.Adam(optimList,weight_decay=w_d)
        else:
            optim = torch.optim.Adam(optimList)
        # optim_name = optim.__class__.__dict__['__module__'].split('.')[-1]
    elif optimizer_name == 'SGD':
        optim = torch.optim.SGD(optimList)
        # optim_name = optim.__class__.__dict__['__module__'].split('.')[-1]
    else:
        print(jutils.toRed('ERR : No Optimizer'))
        raise
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    return optim