import torch
import os

import sys
sys.path.append("..")
from trainer.optim_schedule import ScheduledOptim

model_pt_path = '/home/zhangwentao/host/project/paper/model/left_bert/output/model.pt'

#处理当前的pt文件
model = torch.load(model_pt_path)
model_state = {}
model_state['model_state_dict'] = model['model_state_dict']
state_output_path = os.path.join(os.path.dirname(model_pt_path), 'model_state.pt')
#print(state_output_path)
torch.save(model_state, state_output_path)
