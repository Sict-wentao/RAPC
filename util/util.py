import os
import pandas as pd
import json
import jieba
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style

#加载一个json文件，判断其中得key是不是全都是中文，如果不是则统计输出
def load_json_data(json_file_path):
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    return json_data

#统计输入字典文件中key是否存在英文
def check_en_key(json_file_path):
    json_data = load_json_data(json_file_path)
    for key in json_data:
        #print(key)
        if all('\u4e00' <= char <= '\u9fff' for char in key):
            print(key)

    print('完成')

#将合并当前词典中的token按照jieba指定的格式进行存放
def process_mergedict_to_txt(merge_dict_path, save_folder):
    json_data = load_json_data(merge_dict_path)

    tokens_list = []
    for token in json_data:
        tokens_list.append(token)
    save_file_path = os.path.join(save_folder, 'tokens.txt')
    with open(save_file_path, 'w') as file:
        for token in json_data:
            file.write(token + '\n')

#将合并当前词典中的token，并按照jieba指定的格式进行存放，需要有相对应的频数
def process_mergedict_to_txt_count(merge_dict_path, save_folder):
    json_data = load_json_data(merge_dict_path)

    tokens_list = []
    for token in json_data:
        first_count = sum(json_data[token][0].values())
        second_count = sum(json_data[token][1].values())
        tokens_list.append([token, first_count + second_count])
    
    tokens_list = sorted(tokens_list, key=lambda x: x[1], reverse=True)

    save_file_path = os.path.join(save_folder, 'tokens_count.txt')
    with open(save_file_path, 'w') as file:
        for line in tokens_list:
            line[1] = str(line[1])
            line = ' '.join(line)
            file.write(line + '\n')
    
#将前后缀以及特殊专业词汇词典表和文本切分后的词典表进行合并
def merge_vocab_dict(json_prefix_path, acc_cut_path, merge_folder):
    json_prefix_data = load_json_data(json_prefix_path)
    json_acc_cut_data= load_json_data(acc_cut_path)

    all_key = []
    #将通过jieba切分后的词典文件中的key进行拼音首字母转换，并且保存成和prefix.json一样的结构
    #total = sum(json_acc_cut_data.values())
    acc_all_dict = {}
    for token in json_acc_cut_data:
        #json_acc_cut_data[token] = round(json_acc_cut_data[token] / total, 8)
        pinyin = [char.upper() for char in lazy_pinyin(token, style=Style.FIRST_LETTER)]
        pinyin = ''.join(pinyin)
        if pinyin not in all_key:
            all_key.append(pinyin)
        if pinyin not in acc_all_dict:
            acc_all_dict[pinyin] = {token:json_acc_cut_data[token]}
        else:
            acc_all_dict[pinyin][token] = json_acc_cut_data[token]

    """for key in acc_all_dict:
        #print(key)
        
        total = sum(acc_all_dict[key].values())
        for token in acc_all_dict[key]:
            acc_all_dict[key][token] = round(acc_all_dict[key][token] / total, 8)
        #将该首字母简写下得token按照频率数排序
        acc_all_dict[key] = dict(sorted(acc_all_dict[key].items(), key=lambda item: item[1], reverse=True))"""
    
    #prefix_all_dict = {}
    #将前后缀文件中的词典信息统计成频率并进行排序
    for key in json_prefix_data:
        if key not in all_key:
            all_key.append(key)
        """total = sum(json_prefix_data[key].values())
        for token in json_prefix_data[key]:
            json_prefix_data[key][token] = round(json_prefix_data[key][token] / total, 8)
        json_prefix_data[key] = dict(sorted(json_prefix_data[key].items(), key=lambda item: item[1], reverse=True))"""

    all_key = list(all_key)
    #print(all_key)
    merge_dict = {}
    for key in all_key:
        first_tokens = json_prefix_data[key] if key in json_prefix_data else {}
        second_tokens  = acc_all_dict[key] if key in acc_all_dict else {}
        temp_merge_tokens = [first_tokens, second_tokens]
        merge_dict[key] = temp_merge_tokens

    #将通用的结果保存查看结果
    merge_file_path = os.path.join(merge_folder, 'merge_count.json')
    with open(merge_file_path, 'w') as file:
        json.dump(merge_dict, file, ensure_ascii=False, indent=4)
    

    #print(pinyin, ':', token, ":", json_acc_cut_data[token])
    
    #对前后缀以及专业词进行进一步处理
    #for token in json_prefix_data:


#从存储过程中对中文信息进行抽取，获取中文此表的分词结果
def split_sql_store(sql_store_path, save_folder):
    with open(sql_store_path, 'r') as f:
        contents = f.readlines()
    
    all_split_dict = {}
    acc_split_dict = {}
    #print(type(contents))
    #遍历每一行数据对数据进行分词操作，只将中文分词保存下来
    for line in tqdm(contents, total=len(contents), leave=True, miniters=100):
        temp_all_list = jieba.cut(line, cut_all=True)
        temp_acc_list = jieba.cut(line, cut_all=False)
        temp_all_list = [word for word in temp_all_list if '\u4e00' <= word <= '\u9fff']
        temp_acc_list = [word for word in temp_acc_list if '\u4e00' <= word <= '\u9fff']

        for token in temp_all_list:
            if token not in all_split_dict:
                all_split_dict[token] = 1
            else:
                all_split_dict[token] += 1
        for token in temp_acc_list:
            if token not in acc_split_dict:
                acc_split_dict[token] = 1
            else:
                acc_split_dict[token] += 1
    all_json_path = os.path.join(save_folder, 'all_sql_cut.json')
    acc_json_path = os.path.join(save_folder, 'acc_sql_cut.json')
    with open(all_json_path, 'w') as f:
        json.dump(all_split_dict, f, ensure_ascii=False, indent=4)
    with open(acc_json_path, 'w') as f:
        json.dump(acc_split_dict, f, ensure_ascii=False, indent=4)

#对标注数据进行分词处理，并对分词结果进行统计
def spilt_data(original_file_path, save_folder, HMM=False):
    original_df = pd.read_excel(original_file_path)
    all_data_dict = {}
    acc_data_dict = {}  
    for index, row in original_df.iterrows():
        lables = str(row.loc['中文标注']).strip()
        if len(lables) == 0:
            continue
        #全模式
        all_split_list = jieba.cut(lables, cut_all=True)
        for token in all_split_list:
            if token not in all_data_dict:
                all_data_dict[token] = 1
            else:
                all_data_dict[token] += 1

        #精确模式
        acc_split_list = jieba.cut(lables, cut_all=False)
        for token in acc_split_list:
            if token not in acc_data_dict:
                acc_data_dict[token] = 1
            else:
                acc_data_dict[token] += 1
    #报错切分出来的词典
    all_json_path = os.path.join(save_folder, 'all_cut.json')
    acc_json_path = os.path.join(save_folder, 'acc_cut.json')
    with open(all_json_path, 'w') as f:
        json.dump(all_data_dict, f, ensure_ascii=False, indent=4)
    with open(acc_json_path, 'w') as f:
        json.dump(acc_data_dict, f, ensure_ascii=False, indent=4)

#抽取前缀和次前缀信息
def extract_data(original_file_path, save_folder):
    #抽取前缀信息
    original_df = pd.read_excel(original_file_path)
    prefix_dict = {}
    for index, row in original_df.iterrows():
        lable = str(row.loc['中文标注']).strip()
        first_prefix_name = str(row.loc['前缀']).strip()
        second_prefix_name = str(row.loc['次前缀']).strip()
        if len(lable) == 0 or first_prefix_name == 'nan' or len(first_prefix_name) == 0:
            continue

        lable_list = lable.split('_')
        #print(lable_list)

        if lable_list[0] != 'nan' and len(lable_list[0]) > 0:
            if first_prefix_name not in prefix_dict:
                prefix_dict[first_prefix_name] = {lable_list[0]:1}
            else:
                if lable_list[0] not in prefix_dict[first_prefix_name]:
                    prefix_dict[first_prefix_name][lable_list[0]] = 1
                else:
                    prefix_dict[first_prefix_name][lable_list[0]] += 1

        if len(lable_list) > 1 and len(lable_list[1]) >= len(second_prefix_name) and len(lable_list[1]) > 0:
            if second_prefix_name not in prefix_dict:
                prefix_dict[second_prefix_name] = {lable_list[1][:len(second_prefix_name)]:1}
            else:
                if lable_list[1][:len(second_prefix_name)] not in prefix_dict[second_prefix_name]:
                    prefix_dict[second_prefix_name][lable_list[1][:len(second_prefix_name)]] = 1
                else:
                    prefix_dict[second_prefix_name][lable_list[1][:len(second_prefix_name)]] += 1
    #将该字典保存到一个文件中
    save_json_path = os.path.join(save_folder, 'prefix.json')
    with open(save_json_path, 'w') as file:
        json.dump(prefix_dict, file, ensure_ascii=False, indent=4)

# 对数据进行BLEU指标评估
def evaluate_bleu(label, candidates, n_gram=4, top_k=None, span_matric=False):
    """
        parameters:
            label: 中文字符串
            candidate: 一个列表，里面元素是中文字符串，相当于是对首字母简写翻译之后的候选序列
            n_gram: 选择采用最多n-gram
            top-k: 选择候选项中的top k个结果进行指标计算
        return:
            bleu_scores： bleu计算分数
    """
    if span_matric:
        span = [50, 100]
    reference  = [list(label)]
    if top_k is None:
        top_k = len(candidates)
    candidates = [list(candidate) for candidate in candidates[:top_k]]
    #print(candidates)
    weights = [(1./2., 1./2.),
               (1./3., 1./3., 1./3.),
               (1./4., 1./4., 1./4., 1./4.)]
    chencherry = SmoothingFunction()
    n_gram_bleu_list = [[], [], []]
    for candidate in candidates:
        score = corpus_bleu([reference], [candidate], weights=weights, smoothing_function=chencherry.method1)
        for i in range(len(score)):
            n_gram_bleu_list[i].append(score[i])
        #print(score)
    n_gram_bleu_list = [sum(n_gram)/top_k for n_gram in n_gram_bleu_list]
    return n_gram_bleu_list

# 加载原始测试数据（表名, 中文名），测试结果文件（表名，候选项）
def evaluate_metric(source_path, test_json_path, top_k=50):
    label_df = pd.read_excel(source_path)
    all_evl_result   = []
    bleu_file_name   = test_json_path.split('/')[-1].split('.')[0]+'-bleu'
    bleu_result_path = os.path.join(os.path.dirname(test_json_path), f'{bleu_file_name}.jsonl')
    with open(test_json_path, 'r', encoding='utf-8') as cn_file, open(bleu_result_path, 'w', encoding='utf-8') as bleu_file:
        for line in cn_file.readlines():
            cn_json = json.loads(line)
            letter_name = cn_json['letter_name']
            candidates  = cn_json['cn_name']
            # 获取当前表名对应的中文标签
            label = label_df.loc[label_df['表名'] == letter_name]['中文标注'].iloc[0]
            #print(label.iloc[0])
            # 计算当前样本的bleu值
            temp_bleu_result = evaluate_bleu(label, candidates, top_k=top_k)
            all_evl_result.append(temp_bleu_result)
            # 将当前样本的bleu结果保存到一个文件中
            temp_blue_json = {}
            temp_blue_json['letter_name'] = letter_name
            temp_blue_json['top_k']       = top_k
            temp_blue_json['bleu_result'] = temp_bleu_result
            bleu_file.write(json.dumps(temp_blue_json, ensure_ascii=False) + '\n')
    # 计算所有数据的均值
    all_evl_result = np.array(all_evl_result)
    return np.mean(all_evl_result, axis=0)

def pinyinBleu(source_path, test_folder):
    """
        评估pinyinGPT模型输出结果的bleu指标
    """
    # 获取当前目录下所有的结果文件，分别对其进行评估
    file_list = os.listdir(test_folder)
    for file_name in file_list:
        test_file_path = os.path.join(test_folder, file_name)
        # 获取当前该测试文件中的top k值
        top_k  = file_name.split('.')[0].split('-')[-1]
        result = evaluate_metric(source_path, test_file_path, int(top_k))
        print(f'{top_k}: ', result)

def RLCBleu():
    source_path    = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/all_data/test.xlsx'
    test_json_path = '/home/zhangwentao/host/project/paper-v1.3/data/experiment/v3/small/cot_infer.jsonl'
    for top_k in range(10, 100, 10):
        eva_result = evaluate_metric(source_path, test_json_path, top_k)
        print(top_k, " ", eva_result)


if __name__ == '__main__':
    #json_file_path = '/home/zhangwentao/host/project/information-infer/data/acc_cut.json'
    #check_en_key(json_file_path)
    """source_path    = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/all_data/test.xlsx'
    test_json_path = '/home/zhangwentao/host/project/paper-v1.3/data/experiment/pinyinIME/pinyinConcat-no-fixed'
    pinyinBleu(source_path, test_json_path)"""
    RLCBleu()