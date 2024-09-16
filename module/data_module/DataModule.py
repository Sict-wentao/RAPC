import os
import math
import jieba
import collections
import time
import json
import copy
import tqdm
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from pypinyin import lazy_pinyin, Style

#加载自定义字典
#jieba.load_userdict()

class DataModule:
    def __init__(self, cfg):
        #加载数据库表名文件，其中表/字段名都为待组合数据
        self.cfg = cfg
        self.letter_map_cn = {}
        self.all_list = []
        if self.cfg['custom_jieba'] == True:
            custom_txt_path = os.path.join(self.cfg['custom_letter_path'], 'custom_letter.txt')
            jieba.load_userdict(custom_txt_path)
            self.load_letter_cn_dict()
        # 加载embedding模型,这里主要是用于负采样数据生成，推理阶段不需要
        if 'embedding_model' in self.cfg.keys():
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['embedding_model'], max_length=self.cfg['max_length'])
            self.model = AutoModel.from_pretrained(self.cfg['embedding_model'],).to('cuda:0')
            #print(self.model)
            self.model.eval()
            if torch.cuda.device_count() > 1:
                print("Using %d GPUS for BERT" % torch.cuda.device_count())
                self.model = nn.DataParallel(self.model)
    
    def dag_all(self, cur, cur_list, dag, seq_len):
        if cur not in dag:
            return
        temp_list = dag[cur]
        # 遍历temp_list中的任何一个点作为起始结点
        for i, index in enumerate(temp_list):
            if i == 0:
                now_list = [index]
            else:
                now_list = [temp_list[0], index]
            new_cur_list = copy.deepcopy(cur_list)
            new_cur_list.append(now_list)
            if index == seq_len-1:
                self.all_list.append(new_cur_list)
                return
            self.dag_all(index+1, new_cur_list, dag, seq_len)
        
    #输入一个首字母简写名，对其进行切分并通过首字母简写映射列表进行组合
    def get_letter_cns(self, letter_name):
        # 这里采用全模式进行分词，切分之后将所有的可能组成的分词结果都整合到一起
        self.all_list.clear()
        cn_all_result = []
        letter_dag  = jieba.get_DAG(letter_name)
        self.dag_all(0, [], letter_dag, len(letter_dag))
        for str_list in self.all_list:
            #print(str_list)
            split_str_list = []
            for token in str_list:
                if len(token) == 1:
                    split_str_list.append(letter_name[token[0]])
                else:
                    split_str_list.append(letter_name[token[0]:token[1]+1])
            #print('切分结果: ', split_str_list)
            cn_all_result.extend(self.convert_letter_to_cn(split_str_list))

        return letter_name, cn_all_result

    def get_letter_cn_list(self, letter_name, threashold=0.5, top_k=4):
        """
            describe:
                对首字母简写序列letter_name进行切分，选择长度最短的top_k个切分结果，对切分结果进行中文映射
                返回所有的中文候选项。
            parameter:
                letter_name: 首字母简写
                top_k: 将letter_name进行分词，对分词序列按长度进行排序，选择top k个长度最短的分词序列
            return:
                letter_name: 首字母简写
                cn_all_result: 所有中文候选项
        """
        # 这里采用全模式进行分词，切分之后将所有的可能组成的分词结果都整合到一起
        self.all_list.clear()
        cn_all_result = []
        letter_dag  = jieba.get_DAG(letter_name)
        letter_name_max_len = len(letter_name)-1
        self.dag_all(0, [], letter_dag, len(letter_dag))
        all_split_list = []
        all_split_len  = []
        for str_list in self.all_list:
            split_str_list = []
            clean_len      = 0
            for token in str_list:
                if len(token) == 1:
                    split_str_list.append(letter_name[token[0]])
                    if token != '_':
                        clean_len += 1
                else:
                    split_str_list.append(letter_name[token[0]:token[1]+1])
                    clean_len += 1
            all_split_list.append(split_str_list)
            all_split_len.append(clean_len)
        all_split_len = np.array(all_split_len)
        all_split_len_index = np.argsort(all_split_len)
        # 这里对当前token分词序列进行筛选，分别设置两个阈值
        # (1). len(letter_name)*threahold
        # (2). top_k(如果按照目前阈值计算，没有长度满足，则选择前三个作为候选token序列)
        clean_letter_name = letter_name.strip().replace('_', '')
        #print(clean_letter_name)
        threashold1 = int((len(clean_letter_name)-3)*threashold) + 3
        threashold2 = top_k
        #print("排序后的：",all_split_list)
        for i, index in enumerate(all_split_len_index):
            if all_split_len[index] <= threashold1:
                # 优先根据阈值1进行筛选
                print("one split list: ", all_split_list[index])
                cn_all_result.extend(self.convert_letter_to_cn_list(all_split_list[index]))
            elif all_split_len[0]>threashold1 and i<threashold2:
                # 根据阈值2进行筛选
                print("two split list: ", all_split_list[index])
                cn_all_result.extend(self.convert_letter_to_cn_list(all_split_list[index]))
            else:
                break
            """if i<top_k:
                # 这里可以考虑使用多线程并行处理，可优化
                cn_all_result.extend(self.convert_letter_to_cn_list(all_split_list[index]))"""
        return letter_name, cn_all_result

    def get_all_dict(self, letter_name):
        # 利用当前jieba进行全模式分词处理
        cut_all  = jieba.cut(letter_name, cut_all=True)
        # 查看当前是否切分之后的key是否存在当前映射表，如果存在则将其保留
        all_dict = {}
        for key in cut_all:
            if key in self.letter_map_cn:
                all_dict[key] = self.letter_map_cn[key]
        return all_dict
    
    #提供一个切分之后的首字母序列，从中文映射表内进行检索组合
    def convert_letter_to_cn(self, letter_token_list):
        #先通过一个切分结果进行处理，后续这里需要进行优化
        split_chinese_list = []
        all_result         = []
        #letter_token_list  = list(letter_token_list)
        #print('进来： ', letter_token_list)
        #遍历当前首字母表名切分结果中的首字母简写token
        for letter_token in letter_token_list:
            #print('到这里了')
            temp_list = []
            #如果当前首字母简写token不存在映射表keys中，直接将这个首字母简写token看作中文token
            if letter_token not in self.letter_map_cn.keys():
                #如果当前首字母简写token不存在映射表内，要么就是直接加上去，要么就是直接赋值上去（因为all_result为空）
                if len(all_result) == 0:
                    temp_list.append(letter_token)
                else:
                    for pre_sentence in all_result:
                        #pre_sentence.append(letter_token)
                        temp_list.append(pre_sentence+letter_token)
            else:
                #遍历当前首字母简写token对应的每一个中文映射
                for cn_token in self.letter_map_cn[letter_token]:
                    if len(all_result) == 0:
                        temp_list.append(cn_token)
                    else:
                        for pre_sentence in all_result:
                            temp_list.append(pre_sentence+cn_token)

            #print("到这里: ", temp_list)
            all_result.clear()
            all_result = temp_list
        return all_result
    
    #提供一个切分之后的首字母序列，从中文映射表内进行检索组合,返回的是相对应的list，不是一个完整的字符串
    def convert_letter_to_cn_list(self, letter_token_list): 
        #先通过一个切分结果进行处理，后续这里需要进行优化
        split_chinese_list = []
        all_result         = []
        #letter_token_list  = list(letter_token_list)
        #print('进来： ', letter_token_list)
        #遍历当前首字母表名切分结果中的首字母简写token
        #num = 0
        for letter_token in letter_token_list:
            temp_list = []
            #如果当前首字母简写token不存在映射表keys中，直接将这个首字母简写token看作中文token
            if letter_token not in self.letter_map_cn.keys():
                #如果当前首字母简写token不存在映射表内，要么就是直接加上去，要么就是直接赋值上去（因为all_result为空）
                if len(all_result) == 0:
                    temp_list.append([letter_token])
                else:
                    for pre_sentence in all_result:
                        pre_sentence_temp = copy.deepcopy(pre_sentence)
                        pre_sentence_temp.append(letter_token)
                        temp_list.append(pre_sentence_temp)
            else:
                #遍历当前首字母简写token对应的每一个中文映射
                for cn_token in self.letter_map_cn[letter_token]:
                    if len(all_result) == 0:
                        temp_list.append([cn_token])
                    else:
                        for pre_sentence in all_result:
                            pre_sentence_temp = copy.deepcopy(pre_sentence)
                            pre_sentence_temp.append(cn_token)
                            temp_list.append(pre_sentence_temp)
            all_result.clear()
            all_result = temp_list
        return all_result

    #加载自定义首字母简写到中文的映射字典
    def load_letter_cn_dict(self):
        letter_map_cn_path = os.path.join(self.cfg['letter_map_cn_path'], 'letter_cn.json')
        with open(letter_map_cn_path, 'r') as f:
            self.letter_map_cn = json.load(f)

    #加载指定路径自定义词典
    def load_custom_dict(self):
        custom_letter_path = os.path.join(self.cfg['custom_letter_path'], 'custom_letter.txt')
        jieba.load_userdict(custom_letter_path)

    #读取指定txt文件，利用jieba对该文件进行分词，并将分词结果转换成首字母简写统计其个数到自定义词库文件中
    def generate_custom_dict(self):
        #读取指定文件（txt），并对其进行分词，获取分词结果，并将分词结果统计成自定义词典库
        custom_letter = collections.defaultdict(int)
        letter_map_cn = collections.defaultdict(list)
        with open(self.cfg['field_data_path'], 'r') as file:
            for line in file:
                #对每一行文本数据进行切分，获取其中相应的词典
                line_list = jieba.cut(line, cut_all=True)
                all_cn_list  = [word for word in line_list if '\u4e00' <= word <= '\u9fff']
                
                #将中文转换成拼音首字母
                all_letter_list = [''.join(lazy_pinyin(word, style=Style.FIRST_LETTER)).upper() for word in all_cn_list]
                for index, letter_token in enumerate(all_letter_list):
                    #统计该首字母简写出现的次数
                    custom_letter[letter_token] += 1
                    #将该首字母和中文映射关系添加到映射字典中
                    if letter_token not in letter_map_cn.keys() or all_cn_list[index] not in letter_map_cn[letter_token]:
                        letter_map_cn[letter_token].append(all_cn_list[index])
        
        #将所有映射表存放到一个json文件中
        letter_map_cn_path = os.path.join(self.cfg['letter_map_cn_path'], 'letter_cn.json')
        with open(letter_map_cn_path, 'w') as file:
            json.dump(letter_map_cn, file, ensure_ascii=False, indent=4)
        
        #将统计出来的所有的token都按照：token count进行保存
        custom_letter_path = os.path.join(self.cfg['custom_letter_path'], 'custom_letter.txt')
        with open(custom_letter_path, 'w') as file:
            for key in custom_letter.keys():
                file.write(key + '\n')
    
    # 读取词典文件，将其变成首字母与中文说明映射表
    def genrare_map_dict(self):
        en_cn_map = {}
        with open(self.cfg['key_word_path'], 'r') as file:
            for line in file.readlines():
                line = line.replace('\n','')
                # 转换成大写首字母简写
                en_line = ''.join(lazy_pinyin(line, style=Style.FIRST_LETTER)).upper()
                if en_line not in en_cn_map:
                    en_cn_map[en_line] = [line]
                else:
                    en_cn_map[en_line].append(line)
        letter_map_cn_path = os.path.join(self.cfg['letter_map_cn_path'], 'letter_cn.json')
        with open(letter_map_cn_path, 'w') as file:
            json.dump(en_cn_map, file, ensure_ascii=False, indent=4)
        
        custom_letter_path = os.path.join(self.cfg['custom_letter_path'], 'custom_letter.txt')
        with open(custom_letter_path, 'w') as file:
            for key in en_cn_map.keys():
                file.write(key + '\n')

    # 基于bm25算法找出相关度最高的k个
    def select_top_k_bm25(self, letter_name):
        # 对该表名进行字面相似度计算
        pass

    # 基于bge算法找出语义相似度最高的k个
    def select_top_k_embedding(self, letter_name, label_cn, k=100):
        #print("第一步")
        #torch.cuda.empty_cache()  # 释放显存
        #print("到这里")
        letter_name, cut_all_result = self.get_letter_cn_list(letter_name)
        #print('获取到所有检索结果')
        cut_all_result = [''.join(split_list) for split_list in cut_all_result]
        result_len = len(cut_all_result)
        label_embedding = self.tokenizer(label_cn, padding=True, truncation=True, return_tensors='pt').to('cuda:0')


        # 将cut_all_result根据指定batch size分成多个batch
        with torch.no_grad():
            label_embedding   = self.model(**label_embedding)[0][:, 0]
            # 计算需要分成多少个batch进行预测处理
            batch_size = self.cfg['batch_size']
            num = math.ceil(result_len/batch_size)
            for i in range(num):
                #print('刚进来')
                temp_split_list = cut_all_result[i*batch_size:(i+1)*batch_size]
                #print('i: ', i, " ", len(temp_split_list))
                temp_all_embedding = self.tokenizer(temp_split_list, padding=True, truncation=True, return_tensors='pt').to('cuda:0')
                temp_all_embedding = self.model(**temp_all_embedding)[0][:, 0]
                temp_score = label_embedding @ temp_all_embedding.transpose(0, 1)
                if i == 0:
                    all_score = temp_score
                else:
                    all_score = torch.cat([all_score, temp_score], dim=1)
            #print("all_score shape: ", all_score.shape)
            # 将当前计算结果进行排序，并返回排序后的索引
            all_score = all_score.reshape(-1)
            #print("all_score shape: ", all_score.shape)
            _, indexs = torch.sort(all_score, descending=True)
            #print('indexs shape: ', indexs.shape)
            all_score_len = all_score.shape[0]
            #print("all_score shape: ", all_score.shape[0])
            if k > all_score_len:
                k = all_score_len
            similarity_cn_score = []
            #print('k: ', k)
            for i in range(k//2):
                # 一半存放topk个样本，一半存放最后的topk样本
                similarity_cn_score.append([cut_all_result[indexs[i]], all_score[indexs[i]]])
            for i in range(k//2):
                similarity_cn_score.append([cut_all_result[indexs[all_score_len-1-i]], all_score[indexs[all_score_len-1-i]]])

            #print("里面： ", len(similarity_cn_score))
            del temp_all_embedding
            del all_score
        return similarity_cn_score

    # 读取源文件，批量进行生成top k个分割序列，并生成指定的中文候选项。
    def generate_candidates_json(self, source_path, targe_path):
        # 读取原始excel文件
        source_df = pd.read_excel(source_path)
        all_split_json_path = os.path.join(targe_path, 'all_split.jsonl')
        with open(all_split_json_path, 'w', encoding='utf-8') as file:
            for index,item in source_df.iterrows():
                letter_name = item['表名'].upper()
                letter_names = {}
                letter_names['index'] = index
                letter_names['cn_label'] = item['中文标注']
                letter_names['letter_name'] = letter_name
                # 进行中文切分，获取多个
                self.all_list.clear()
                letter_dag  = jieba.get_DAG(letter_name)
                self.dag_all(0, [], letter_dag, len(letter_dag))
                all_split_list = []
                all_split_len  = []
                for str_list in self.all_list:
                    split_str_list = []
                    clean_len      = 0
                    for token in str_list:
                        if len(token) == 1:
                            split_str_list.append(letter_name[token[0]])
                            if token != '_':
                                clean_len += 1
                        else:
                            split_str_list.append(letter_name[token[0]:token[1]+1])
                            clean_len += 1
                    all_split_list.append(split_str_list)
                    all_split_len.append(clean_len)
                #print('###############', all_split_list)
                all_split_len = np.array(all_split_len)
                all_split_len_index = np.argsort(all_split_len)
                candidate_token_list = []
                all_cn_dict = {}
                # 获取排序后的前top_k个
                for i, sort_index in enumerate(all_split_len_index):
                    if i<self.cfg['top_k_split']:
                        # 这里可以考虑使用多线程并行处理，可优化
                        candidate_token_list.append(all_split_list[sort_index])
                        # 获取当前token分词结果的所有中文映射候选项
                        cn_candidate_list = self.convert_letter_to_cn_list(all_split_list[sort_index])
                        cn_candidate_list = [''.join(cn_item) for cn_item in cn_candidate_list]
                        all_cn_dict[i] = cn_candidate_list
                letter_names['split_token_list'] = candidate_token_list
                # 将结果写入到json文件中
                targe_json_folder = os.path.join(targe_path, 'cn_data', f'{index}')
                # 判断当前目录是否存在，不存在则创建该文件
                if not os.path.exists(targe_json_folder):
                    os.makedirs(targe_json_folder)
                targe_json_path = os.path.join(targe_json_folder, 'cn_list.json')
                with open(targe_json_path, 'w', encoding='utf-8') as tfile:
                    json.dump(all_cn_dict, tfile, ensure_ascii=False)
                file.write(json.dumps(letter_names, ensure_ascii=False) + '\n')

    def embedding_cn(self, all_json_path):
        # 读取json文件，处理其中每一条数据
        scale = 'small'
        data_folder_path = os.path.dirname(all_json_path)
        #print(data_folder_path)
        row_index = 0
        with open(all_json_path, 'r', encoding='utf-8') as all_file:
            for line_data in all_file.readlines():
                if row_index < 433:
                    row_index += 1
                    continue
                print(row_index)
                row_index += 1
                data = json.loads(line_data)
                # 获取当前样本的index，再根据index找到与之对应生成的中文候选项json文件
                index = data['index']
                cn_json_folder = os.path.join(data_folder_path, 'cn_data', f'{index}')
                cn_json_path   = os.path.join(cn_json_folder, 'cn_list.json')
                with open(cn_json_path, 'r', encoding='utf-8') as cn_json_file:
                    cn_json_data = json.load(cn_json_file)
                # 遍历每一个id对应的中文候选项，将其拼接到一起，对其中全部候选项进行embedding
                all_cn_list = []
                cn_json_num = len(cn_json_data)
                for i in range(cn_json_num):
                    all_cn_list.extend(cn_json_data[str(i)])

                # 调用embedding模型进行向量化，分批次进行向量化
                all_cn_len = len(all_cn_list)
                batch_size = self.cfg['batch_size']
                num = math.ceil(all_cn_len/batch_size)
                for i in range(num):
                    temp_split_list = all_cn_list[i*batch_size:(i+1)*batch_size]
                    #print('i: ', i, " ", len(temp_split_list))
                    temp_all_embedding = self.tokenizer(temp_split_list, padding=True, truncation=True, return_tensors='pt').to('cuda:0')
                    with torch.no_grad():
                        temp_all_embedding = self.model(**temp_all_embedding)[0][:, 0]
                    temp_all_embedding = F.normalize(temp_all_embedding, p=2, dim=-1).to('cpu')
                    if i == 0:
                        all_embeddings = temp_all_embedding
                    else:
                        all_embeddings = torch.cat([all_embeddings, temp_all_embedding], dim=0)
                    #print('all_embeddings shape: ', all_embeddings.shape)
                # 将数据按照index进行保存
                start_index = 0
                for i in range(cn_json_num):
                    temp_len  = len(cn_json_data[str(i)])
                    end_index = start_index+temp_len
                    tensor_data = all_embeddings[start_index:end_index]
                    start_index = end_index
                    # 保存tensor_data
                    #print('i: ', i)
                    #print('temp_len: ', temp_len)
                    #print('tensor_data shape: ', tensor_data.shape)
                    tensor_folder = os.path.join(cn_json_folder, scale, str(i))
                    if not os.path.exists(tensor_folder):
                        os.makedirs(tensor_folder)
                    tensor_path = os.path.join(tensor_folder, 'embedding.pt')
                    torch.save(tensor_data, tensor_path)
                del all_embeddings
                #break


def main():
    #测试该模块是否正常运行
    cfg = {}
    #cfg['field_data_path'] = '/home/zhangwentao/host/project/paper/data/vocab/sql_store.txt'
    cfg['key_word_path']   = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/key_word/keyWord.txt'
    cfg['letter_map_cn_path'] = '/home/zhangwentao/host/project/paper-v1.3/data/vocab'
    cfg['custom_letter_path'] = '/home/zhangwentao/host/project/paper-v1.3/data/vocab'
    cfg['top_k_split'] = 8
    cfg['batch_size'] = 40960
    cfg['max_length'] = 512
    #cfg['embedding_model'] = '/home/zhangwentao/host/project/paper-v1.3/module/twin_module/model/model/right_bert/bge-small-zh-v1.5'
    cfg['custom_jieba']  = True
    dataTool = DataModule(cfg)
    #dataTool.genrare_map_dict()

    #随机输入一个首字母简写部分，测试中文全称组合
    """letter_name, cut_all_result = dataTool.get_letter_cn_list('BQ_YJCXDLBQDYK', threashold=0.5, top_k=cfg['top_k_split'])
    print(len(cut_all_result))"""
    #label = ['病区','_', '医生', '_', '病案', '手术', '库']
    #print(cut_all_result[:10])
    #print("结果： ", len(cut_all_result))
    #print("表名： ", letter_name)
    #print('label: ', label in cut_all_result)

    # 计算向量相似度
    letter_name = 'BQ_YJCXDLBQDYK'
    label = '病区_医生_病案手术库'
    similarity_cn_score = dataTool.select_top_k_embedding(letter_name, label, k=200)
    print(similarity_cn_score)

    """source_file_path = '/home/zhangwentao/host/project/paper-v1.3/data/experiment/train.xlsx'
    target_path      = '/home/zhangwentao/host/project/paper-v1.3/data/experiment'
    dataTool.generate_candidates_json(source_file_path, target_path)"""

    """all_json_path = '/home/zhangwentao/host/project/paper-v1.3/data/experiment/all_split.jsonl'
    dataTool.embedding_cn(all_json_path)"""


if __name__ == '__main__':
    main()




