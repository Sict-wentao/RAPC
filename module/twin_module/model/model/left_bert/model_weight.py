import torch
import os

# 只保留训练之后模型中的bert部分的模型参数
def model_weight_process(weight_path, target_path):
    # 读取当前权重文件
    target_path = os.path.join(target_path, 'left_bert.pt')
    new_state = {}
    state = torch.load(weight_path)
    new_state['bert_model_state'] = state['bert_model_state_dict']
    # 将清洗后的权重文件，保存到指定路径下
    torch.save(new_state, target_path)

if __name__ == "__main__":
    weight_path = '/home/zhangwentao/host/project/paper/model/model/left_bert/output/run-0530/epoch_19/checkpoint_19_6999.pt'
    target_path = '/home/zhangwentao/host/project/paper/model/output'
    model_weight_process(weight_path, target_path)