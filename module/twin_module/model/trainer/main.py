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
from trainer import TwinTowerTrainer


def pre_train(hpy='', opt='', device=''):
    #------------------------------超参数---------------------------------------------

    #初始化bert模型结构参数，可以将这一部分的参数写入到一个yaml文件中
    left_model_path = '/home/zhangwentao/host/project/paper-v1.3/module/twin_module/model/model/left_bert'
    left_model_weight_path = '/home/zhangwentao/host/project/paper-v1.3/module/twin_module/model/output/left_bert.pt'
    right_model_path = '/home/zhangwentao/host/project/paper-v1.3/module/twin_module/model/model/right_bert/bge-large-zh-v1.5'
    train_path = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/train_data/v2/train-v2.txt'
    test_path  = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/train_data/v2/test-v2.txt'
    #train_path  = '/home/zhangwentao/host/project/paper-v1.3/data/TwinTower/train.txt'
    
    # 双塔模型权重
    #mergeNet_weight_path = '/home/zhangwentao/host/project/paper-v1.3/module/twin_module/model/output/weight-large-0615-pretrain/step_best.pt'
    #mergeNet_weight_path = '/home/zhangwentao/host/project/paper-v1.3/module/twin_module/model/output/weight/step_last.pt'
    mergeNet_weight_path  = None

    left_hidden_size = 768
    n_layers = 12
    attn_heads = 12
    seq_len = 128

    hidden_size = 1024
    
    batch_size = 4096

    epochs = 60
    lr = 1e-4

    print("Loading Left Vocab", left_model_path)
    left_model_vocab_path = os.path.join(left_model_path, 'dataset', 'vocab.pkl')
    left_vocab  = WordVocab.load_vocab(left_model_vocab_path)

    print('Left Vocab Size: ', len(left_vocab))
    print('left vocab: ', left_vocab.stoi)
    print('Initialize left model....')
    left_model = Left_model(len(left_vocab), left_hidden_size, n_layers, attn_heads)
    #加载模型权重文件
    print('left weight path: ', left_model_weight_path)
    left_model_weight = torch.load(left_model_weight_path)
    print('left weight load secc...')

    #初始化模型权重
    left_model.load_state_dict(left_model_weight['bert_model_state'])
    left_model.eval()

    print("Loading Right Model... :", right_model_path)
    right_tokenizer = AutoTokenizer.from_pretrained(right_model_path)
    right_model = AutoModel.from_pretrained(right_model_path)
    right_model.eval()


    print('Loading Train Dataset ...')
    train_dataset = TwinTowerDataset(train_path, 
                                     left_vocab=left_vocab,
                                     right_vocab=right_tokenizer,
                                     seq_len=seq_len,
                                     pretrain=False)
    
    print('Loading Test Dataset ...')
    test_dataset = TwinTowerDataset(test_path, 
                                     left_vocab=left_vocab,
                                     right_vocab=right_tokenizer,
                                     seq_len=seq_len,
                                     pretrain=False)
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_dataset.get_batch_items)
    test_data_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=test_dataset.get_batch_items)
    #model = TwinTower(hidden_size=hidden_size, left_model=left_model, right_model=right_model)

    #------------------------------------------------------------------------------------------

    #--------------------------------------定义模型训练器---------------------------------------
    trainer = TwinTowerTrainer(left_model=left_model,
                               right_model=right_model,
                               hidden_size=hidden_size,
                               train_dataloader=train_data_loader,
                               test_dataloader=test_data_loader,
                               mergeNet_weight_path=mergeNet_weight_path,
                               lr=lr)

    #这里是进行迭代训练的代码
    print('start train TwinTowerModel...')
    for epoch in range(epochs):
        trainer.train(epoch)
        trainer.test(epoch)

if __name__ == '__main__':
    pre_train()


