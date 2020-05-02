import os
import torch
import jutils
import TOTAL_Prepare

def compare_test_name(input_dic):
    vis = jutils.Visdom(name = input_dic['vis_name'],port=7097)
    for stn_type in os.listdir(input_dic['test_name']):
        print(jutils.toMagenta('STN_Type        : '+ stn_type))
        test_loss_list = {}
        test_iter_list = {}
        for warping_type in os.listdir(os.path.join(input_dic['test_name'],stn_type)):
            tuning_loss_list={}
            tuning_best_loss={}
            tuning_iter_list = {}
            for tuning_type in os.listdir(os.path.join(input_dic['test_name'],stn_type,warping_type)):
                project_path = os.path.join(input_dic['test_name'],stn_type,warping_type,tuning_type)
                max_val = TOTAL_Prepare.get_last_weight(project_path)
                if max_val is not None:
                    net_params = torch.load(os.path.join(project_path, 'weight', 'weight_'+str(max_val)+'.pth'))
                    tuning_loss_list[warping_type + '(' + tuning_type + ')'] = net_params['test_err_list']
                    tuning_best_loss[warping_type+'('+tuning_type+')'] = min(net_params['test_err_list'])
                    tuning_iter_list[warping_type + '(' + tuning_type + ')'] = net_params['iter']

            if (input_dic['compare_option']=='all_tuning'):
                test_loss_list.update(tuning_loss_list)
                test_iter_list.update(tuning_iter_list)
            elif(input_dic['compare_option'] == 'best_tuning'):
                test_loss_list.update({k: v for k, v in tuning_loss_list.items() if min(v) == min(tuning_best_loss.values())})
                test_iter_list.update(tuning_iter_list)
        win = None
        for k, v in test_loss_list.items():
            win = vis.Eval_plot(k,stn_type,test_iter_list[k],v,win)


eval_option={}
eval_option['test_name'] = 'Results/TESTNoPert'
eval_option['vis_name'] = 'compare'
eval_option['compare_option'] = 'all_tuning' # Option : 'all_tuning' / 'best_tuning'
compare_test_name(eval_option)