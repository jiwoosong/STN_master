# STN_master
This project provides the total solution of STN, the most input is defined by dictionary input form.
## Caution
- PlEASE Don't delete any FILES...!
- PLEASE Checkout .gitignore and careful when push files...! 
(possible but, it's hard to return past repository...)
- PLEASE naming filenames clear and carefully.
- PLEASE USE personal visdom port.
- PLEASE Don't upload weight files(large datas).
## Experiments
- No Perts & STN_enforce

| warper | classifer | glr/clr/wd | acc | options |
|:---:|:---:|:---:|:---:|:---:|
|`Bilinear`|`FCLayer`|1e-5/1e-3/1e-4|0.0%||
|`Bilinear`|`FCLayer`|1e-4/1e-3/1e-4|0.0%||
|`Bilinear`|`FCLayer`|1e-3/1e-3/1e-4|0.0%||

- No Perts & ICSTN_enforce

| warper | classifer | glr/clr/wd | acc | options |
|:---:|:---:|:---:|:---:|:---:|
|`Bilinear`|`FCLayer`|1e-5/1e-3/1e-4|0.0%||
|`Bilinear`|`FCLayer`|1e-4/1e-3/1e-4|0.0%||
|`Bilinear`|`FCLayer`|1e-3/1e-3/1e-4|0.0%||

- Perts & STN_enforce

| warper | classifer | glr/clr/wd | acc | options |
|:---:|:---:|:---:|:---:|:---:|
|`Bilinear`|`FCLayer`|1e-5/1e-3/1e-4|0.0%||
|`Bilinear`|`FCLayer`|1e-4/1e-3/1e-4|0.0%||
|`Bilinear`|`FCLayer`|1e-3/1e-3/1e-4|0.0%||

- Perts & ICSTN_enforce

| warper | classifer | glr/clr/wd | acc | options |
|:---:|:---:|:---:|:---:|:---:|
|`Bilinear`|`FCLayer`|1e-5/1e-3/1e-4|0.0%||
|`Bilinear`|`FCLayer`|1e-4/1e-3/1e-4|0.0%||
|`Bilinear`|`FCLayer`|1e-3/1e-3/1e-4|0.0%||

## Usage
Create 'gpu0.py' in the project page 
``` Python
import os
import torch
from TOTAL_Prepare import Prepare_Training2
from TOTAL_MAIN import training

os.environ["CUDA_DEIVCES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "?"
W_in = 50
H_in = 50
W_warp = 50
H_warp = 50

from Networks.STN_enforce.STNenforce import STN
from Networks.warper.Basic.Bilinear import Bilinear
from Networks.classifier.FClayer import Classification_net

warper = Bilinear(align_corners=True)
geometric = STN(W_warp=W_warp, H_warp=H_warp, transformImage=warper)
classifier = Classification_net(W_warp=W_warp, H_warp=H_warp, chanel=3)
prepare_input={}
prepare_input['test_name'] = 'Results/?'
prepare_input['message'] = ''
prepare_input['vis_port'] = ?
prepare_input['geometric'] = geometric
prepare_input['classifier'] = classifier
prepare_input['glr'] = ?
prepare_input['clr'] = ?
prepare_input['wd'] = ?
prepare_input['optimizer_name'] = '?'
prepare_input['criterion'] = torch.nn.CrossEntropyLoss()
prepare_input['load_flag'] = ?
prepare_input['batch_size'] = 64
# Maximum iter / Validation iter / Save iter
prepare_input['max_iter'] = 300000
prepare_input['val_iter'] = 1000
prepare_input['save_iter'] = [(1000,100),(10000,1000),(prepare_input['max_iter'],20000)] #[(iter, period)...]
prepare_dic = Prepare_Training2(prepare_input)
training(prepare_dic, W_in=W_in, H_in=H_in)
```
