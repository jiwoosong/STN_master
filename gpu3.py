from TOTAL_Prepare import Prepare_Training2
from TOTAL_MAIN import training
import os
import torch
os.environ["CUDA_DEIVCES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
W_in = 50
H_in = 50
W_warp = 50
H_warp = 50
from Networks.STN_enforce.STNenforce import STN
from Networks.warper.Basic.Bilinear import Bilinear
from Networks.classifier.Global_weight_AP import Classification_net


warper = Bilinear(align_corners=True)
geometric = STN(W_warp=W_warp, H_warp=H_warp, transformImage=warper)
classifier = Classification_net(in_chanel=3, out_chanel=43)
prepare_input={}
prepare_input['test_name'] = 'Results/TESTNoPert'
prepare_input['vis_port'] = 8097
prepare_input['geometric'] = geometric
prepare_input['classifier'] = classifier
prepare_input['glr'] = 1e-5
prepare_input['clr'] = 1e-3
prepare_input['wd'] = 1e-4
prepare_input['optimizer_name'] = 'Adam'
prepare_input['criterion'] = torch.nn.CrossEntropyLoss()
prepare_input['load_flag'] = False
prepare_input['batch_size'] = 64
# Maximum iter / Validation iter / Save iter
prepare_input['max_iter'] = 300000
prepare_input['val_iter'] = 1000
prepare_input['save_iter'] = [(1000,100),(10000,1000),(prepare_input['max_iter'],20000)] #[(iter, period)...]
prepare_dic = Prepare_Training2(prepare_input)
training(prepare_dic, W_in=W_in, H_in=H_in)