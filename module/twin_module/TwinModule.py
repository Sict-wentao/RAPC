import os
import torch
import jieba
import time
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from transformers import AutoTokenizer, AutoModel

import sys 
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-1])
# print('双塔模块：', root_path)
sys.path.append(root_path)

from model.model.TwinTower import TwinTower as model
from model.dataset.vocab import WordVocab
from model.model.left_bert.model import BERT as Left_model

class TwinModule():
    def __init__(self, cfg):
        #初始化模型
        self.cfg = cfg
        left_model_vocab_path = os.path.join(self.cfg['left_model_path'], 'dataset', 'vocab.pkl')
        print("left_model vocab path: ", left_model_vocab_path)
        self.left_vocab = WordVocab.load_vocab(left_model_vocab_path)
        print('Loading Left Model...')
        self.left_model = Left_model(len(self.left_vocab),
                                     self.cfg['left_hidden_size'],
                                     self.cfg['n_layers'],
                                     self.cfg['attn_heads'])
        #加载left模型权重文件
        left_model_weight      = torch.load(self.cfg['left_model_weight_path'])
        #初始化模型权重
        self.left_model.load_state_dict(left_model_weight['bert_model_state'])

        print("Loading Right Model...")
        self.right_tokenizer = AutoTokenizer.from_pretrained(self.cfg['right_model_path'])
        self.right_model     = AutoModel.from_pretrained(self.cfg['right_model_path'])

        print("Loading TwinTower")
        self.model = model(hidden_size=self.cfg['hidden_size'], 
                            left_model=self.left_model, 
                            right_model=self.right_model,
                            mergeNet_layer=self.cfg['mergeNet_layer']).to(self.cfg['device'])
        
        print("Loading TwinTower Weight...")
        model_state_dict = torch.load(self.cfg['TwinTower_model_weight_path'])
        self.model.mergeNet.load_state_dict(model_state_dict['mergeNet_state_dict'])
        self.model.eval()
        print("Loading model seccessful....")

        # 分布式推理, 后面需要分开对双塔模型进行处理
        """if torch.cuda.device_count() > 1:
            print("Using %d GPUS for Model" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
        self.model.to(self.cfg['device'])"""

        self.left_model_device  = torch.device('cpu')
        self.right_model_device = torch.device(self.cfg['device'])  
        self.mergenet_device    = torch.device('cuda:3')     

        if torch.cuda.device_count() > 1:
            left_model_devices = [0, 1, 2]
            print("Using %d GPUS for Model" % torch.cuda.device_count())
            self.right_model = nn.DataParallel(self.model.right_model, device_ids=left_model_devices)
        self.right_model.to(self.right_model_device)

        self.model.mergeNet.to(self.mergenet_device)

        self.model.left_model.to(self.left_model_device)
    
    def predict(self, inputs):
        # 将当前输入数据中的表名和多个候选中文全称进行分别输入到left_model和right_mdoel
        left_input, all_pt_rihgt_inputs = self.get_pt_input(inputs)
        all_output = torch.tensor([])
        # left_input：input_ids (1, seq_len), segment_label(1, seq_len)
        # all_pt_right_inputs: 为一个列表，列表中的元素为分批次处理的right_input: input_ids, token_type_ids, attention_mask
        #-----------------------------
        # 优化处理left model处理，这里的left input只需要输出一次即可，后续将left model输出结果直接和right model输出结果进行拼接即可，减少计算量
        # 先处理left model，获得left embedding
        left_input = {key: value.to(self.left_model_device) for key, value in left_input.items()}
        left_embedding = self.model.left_model(left_input['input_ids'], left_input['segment_label'])[:, 0]
        left_embedding  = F.normalize(left_embedding, p=2, dim=-1).to(self.left_model_device)
        # 处理right model， 获得right embedding, 由于需要进行embedding向量化数据较多
        # 分批次进行处理
        for index in trange(len(all_pt_rihgt_inputs)):
            right_inputs = all_pt_rihgt_inputs[index]
            right_inputs = {key: value.to(self.cfg['device']) for key, value in right_inputs.items()}
            # 将self.right模型分布式推理
            with torch.no_grad():
                right_embedding = self.right_model(**right_inputs)[0][:, 0]
            right_embedding     = F.normalize(right_embedding, p=2, dim=-1).to(self.mergenet_device)
            # 将数据维度扩充到和right_embedding一样的batch_size
            batch_size          = right_embedding.shape[0]
            left_embedding_twin = left_embedding.repeat(batch_size, 1).to(self.mergenet_device)
            with torch.no_grad():
                #mergenet_start_time = time.time()
                output = self.model.mergeNet(left_embedding_twin, right_embedding)
                #print(index)
                output = torch.softmax(output, dim=-1).to('cpu')[..., 1]
                #print(time.time()-mergenet_start_time)
            all_output = torch.cat((all_output, output), dim=0)
            del right_embedding
            del left_embedding_twin

        sort_output, sort_index = all_output.sort(descending=True)
        return sort_output, sort_index
    
    def predict_old(self, inputs):
        # 将当前输入数据中的表名和多个候选中文全称进行分别输入到left_model和right_mdoel
        left_input, all_pt_rihgt_inputs = self.get_pt_input(inputs)
        first_left_input  = {}
        second_left_input = {}
        all_output = torch.tensor([])
        # left_input：input_ids (1, seq_len), segment_label(1, seq_len)
        # all_pt_right_inputs: 为一个列表，列表中的元素为分批次处理的right_input: input_ids, token_type_ids, attention_mask
        right_batch_size = all_pt_rihgt_inputs[0]['input_ids'].shape
        #print('原始input_ids shape: ', left_input['input_ids'].shape)
        #-----------------------------
        # 优化处理left model处理，这里的left input只需要输出一次即可，后续将left model输出结果直接和right model输出结果进行拼接即可，减少计算量
        #-----------------------------
        first_left_input['input_ids']     = left_input['input_ids'].repeat(right_batch_size[0], 1)
        first_left_input['segment_label'] = left_input['segment_label'].repeat(right_batch_size[0], 1)
        first_left_input = {key: value.to(self.cfg['device']) for key, value in first_left_input.items()}

        if all_pt_rihgt_inputs[0]['input_ids'].shape != all_pt_rihgt_inputs[-1]['input_ids'].shape:
            # 需要针对最后一个rihgt_inputs的shape进行left_input定制
            right_batch_size = all_pt_rihgt_inputs[-1]['input_ids'].shape
            second_left_input['input_ids']     = left_input['input_ids'].repeat(right_batch_size[0], 1)
            second_left_input['segment_label'] = left_input['segment_label'].repeat(right_batch_size[0], 1)
            second_left_input = {key: value.to(self.cfg['device']) for key, value in second_left_input.items()}
        for index in trange(len(all_pt_rihgt_inputs)):
            right_inputs = all_pt_rihgt_inputs[index]
            right_inputs = {key: value.to(self.cfg['device']) for key, value in right_inputs.items()}
            # 将数据分批输入到模型中
            if index == len(all_pt_rihgt_inputs)-1 and len(second_left_input)>0:
                # 特殊处理
                with torch.no_grad():
                    output = self.model(second_left_input, right_inputs).to('cpu')[..., 1]
            else:
                #print('left_model shape: ', first_left_input['input_ids'].shape)
                #print('right_model shape: ', right_inputs['input_ids'].shape)
                with torch.no_grad():
                    output = self.model(first_left_input, right_inputs).to('cpu')[..., 1]
            #将模型输出结果添加到一个列表中
            all_output = torch.cat((all_output, output), dim=0)
            del right_inputs
            # 清空显存
            #torch.cuda.empty_cache()
            #time.sleep(10)
        # 将模型输出结果排序，并返回排序索引
        sort_output, sort_index = all_output.sort(descending=True)
        #print('原始输出： ', all_output)
        #print('排序后的输出： ', sort_output)
        #print('排序后的索引： ', sort_index)
        return sort_output, sort_index

    def get_pt_input(self, inputs):
        #通过将输入原始数据转换成一个tensor数据，直接满足模型输入格式要求
        #假设这里进来以后就都是token list
        #left_model
        left_inputs = {}
        #1.将input['left_input']进行tokenizer
        letter_name = inputs['letter_name']
        # 这里需要补充对首字母简写表名进行切分操作
        left_token_list = list(letter_name)
        left_token_ids     = [self.left_vocab.stoi.get(token, self.left_vocab.unk_index) for token in left_token_list]
        left_segment_label = [1 for _ in range(len(left_token_ids))]
        left_padding       = [self.left_vocab.pad_index for _ in range(self.cfg['seq_len']-len(left_token_ids))]

        left_token_ids.extend(left_padding), left_segment_label.extend(left_padding)
        left_input_ids     = torch.tensor([left_token_ids[:self.cfg['seq_len']]])
        left_segment_label = torch.tensor([[1 for _ in range(len(left_input_ids))]])
        left_inputs['input_ids']     = left_input_ids
        left_inputs['segment_label'] = left_segment_label
        #print('最开始的input_ids shape: ', left_inputs['input_ids'].shape)
        #left_mdoel需要两个输入，一个是input_ids,另一个是segment_label
        # 需要加载自定义的词汇库

        #right_model
        #1.同时将多个中文候选项tokenizer
        #right_inputs: input_ids, token_type_ids, attention_mask
        batch_cn_results = [inputs['cn_all_results'][i:i+self.cfg['twin_batch_size']]
                            for i in range(0, len(inputs['cn_all_results']), self.cfg['twin_batch_size'])]
        all_pt_rihgt_inputs = []
        #left的所有的数据都
        #分批进行处理，并将模型输出结果都给保存起来
        for cn_results in batch_cn_results:
            #print(cn_results[:2])
            right_inputs = self.right_tokenizer(cn_results, 
                                                max_length=self.cfg['seq_len'], 
                                                padding='max_length', 
                                                truncation=True, 
                                                return_tensors='pt')
            #print(right_inputs.keys())
            all_pt_rihgt_inputs.append(right_inputs)
            # 这里需要将候选数据分批进行处理，否则数据量太大了，自定义一个数据加载器
            # 这里的right_inputs['input_ids'].shape == (batch_size, )
            #2.将每个token list都转换成tensor
        return left_inputs, all_pt_rihgt_inputs

if __name__ == '__main__':
    cfg = {}
    cfg['left_model_vocab_path'] = '/home/zhangwentao/host/project/paper/module/twin_module/model/model/left_bert'
    cfg['left_vocab_size']       = 20005
    cfg['left_hidden_size']      = 512
    cfg['n_layers']              = 12
    cfg['attn_heads']            = 8
    cfg['left_model_path']       = '/home/zhangwentao/host/project/paper/module/twin_module/model/model/left_bert'

    cfg['seq_len'] = 80

    cfg['right_model_path']      = '/home/zhangwentao/host/project/paper/module/twin_module/model/model/right_bert/bge-large-zh-v1.5'

    cfg['TwinTower_model_weight_path'] = '/home/zhangwentao/host/project/paper/module/twin_module/model/output/weight/last.pt'
    cfg['hidden_size']           = 512
    cfg['twin_batch_size']       = 2

    cfg['device'] = 'cuda'

    twinmodule = TwinModule(cfg)
    inputs = {
        'letter_name':'LJ_PGZXZBMXK',
        'cn_all_results':['逻辑品管执行者编码新科', 
                          '连接品管执行者编码新科', 
                          '累计品管执行者编码新科', 
                          '累积品管执行者编码新科', 
                          '理解品管执行者编码新科', 
                          '垃圾品管执行者编码新科', 
                          '立即品管执行者编码新科', 
                          '了解品管执行者编码新科', 
                          '龙江品管执行者编码新科']
    }
    twinmodule.predict(inputs)
