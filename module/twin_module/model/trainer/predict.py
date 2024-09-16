import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel

import sys
sys.path.append("..")

from dataset.vocab import WordVocab
from model.left_bert.model import BERT as Left_model
from dataset.dataset import TwinTowerDataset
from model.TwinTower import TwinTower

def predict():
    left_model_path   = '/home/zhangwentao/host/project/paper/model/model/left_bert'
    right_model_path  = '/home/zhangwentao/host/project/paper/model/model/right_bert/bge-large-zh-v1.5'
    corpus_path       = '/home/zhangwentao/host/project/paper/data/TwinTower/train-500.txt'
    model_weight_path = '/home/zhangwentao/host/project/paper/model/output/weight/last.pt'
    test_log_path     = '/home/zhangwentao/host/project/paper/model/output/test.txt'

    left_vocab_size = 20005
    left_hidden_size = 512
    n_layers   = 12
    attn_heads = 8

    hidden_size = 512
    seq_len     = 80

    batch_size = 16

    device   = 'cuda:0'
    log_freq = 10

    print("Loading Left Vocab", left_model_path)
    left_model_vocab_path = os.path.join(left_model_path, 'output', 'vocab.pkl')
    left_vocab  = WordVocab.load_vocab(left_model_vocab_path)
    #print('Left Vocab Size: ', len(left_vocab))
    print('Initialize left model....')
    left_model = Left_model(left_vocab_size, left_hidden_size, n_layers, attn_heads)
    #加载模型权重文件
    left_model_weight_path = os.path.join(left_model_path, 'output', 'model_state.pt')
    left_model_weight = torch.load(left_model_weight_path)

    #初始化模型权重
    left_model.load_state_dict(left_model_weight['model_state_dict'][0])
    #left_model.eval()

    print("Loading Right Model... :", right_model_path)
    right_tokenizer = AutoTokenizer.from_pretrained(right_model_path)
    right_model = AutoModel.from_pretrained(right_model_path)
    #right_model.eval()

    print('Loading test Dataset ...')
    test_dataset = TwinTowerDataset(corpus_path, 
                                     left_vocab=left_vocab,
                                     right_vocab=right_tokenizer,
                                     seq_len=seq_len)
    
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, collate_fn=test_dataset.get_batch_items)
    model = TwinTower(hidden_size=hidden_size, left_model=left_model, right_model=right_model).to(device)

    print("加载权重之前 left_model.transformer_blocks.9.feed_forward.w_1.bias weight", model.state_dict()['left_model.transformer_blocks.9.feed_forward.w_1.bias'])
    print("模型加载之前初始化的值 : mergeNet.left_layer.bias", model.state_dict()['mergeNet.left_layer.bias'])
    
    #加载模型后面的参数权重
    model_state_dict = torch.load(model_weight_path)
    print("权重文件中对应的值: mergeNet.left_layer.bias", model_state_dict['model_state_dict']['mergeNet.left_layer.bias'])
    model.load_state_dict(model_state_dict['model_state_dict'], strict=False)
    model.eval()
    #print("双塔模型权重键: ", model_state_dict['model_state_dict'].keys())
    print("加载之后的值: mergeNet.left_layer.bias", model.state_dict()['mergeNet.left_layer.bias'])
    print("加载模型之后 left_model.transformer_blocks.9.feed_forward.w_1.bias weight", model.state_dict()['left_model.transformer_blocks.9.feed_forward.w_1.bias'])

    #初始化损失函数
    loss_func = F.binary_cross_entropy
    avg_loss  = 0
    total_correct = 0
    total_element = 0

    #预测数据
    for i, data in enumerate(test_data_loader):
        input_data = {}
        for key, value in data.items():
            if key not in input_data.keys():
                input_data[key] = {}
            if type(value)==dict:
                for sub_key, sub_value in data[key].items():
                    input_data[key][sub_key] = data[key][sub_key].to(device)
            else:
                input_data[key] = data[key].to(device)
        
        output  = model.forward(input_data["left_input"], input_data["right_input"])
        loss    = loss_func(output, input_data['label'])
        output  = torch.where(output>0.5, torch.ones_like(output), torch.zeros_like(output))
        correct = output.eq(input_data['label']).sum().item()

        avg_loss      += loss.item()
        total_correct += correct
        total_element += input_data['label'].nelement()

        post_fix = {
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

        if i % log_freq == 0:
            #data_iter.write(str(post_fix))
            #将当前训练模型的结果保存到指定的文件中
            with open(test_log_path, 'a', encoding='utf-8') as file:
                file.write(str(post_fix) + '\n')

if __name__ == '__main__':
    predict()



