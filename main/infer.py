import os
import pandas as pd
import json
import jieba
import re
import copy
from tqdm import tqdm
import time
import random
from pypinyin import lazy_pinyin
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
import threading

import sys
sys.path.append("..")
from config.prompt import example_tamplete, instruction_examples, select_tamplete, select_only_one_tamplete
from util.util import extract_data, spilt_data, split_sql_store, merge_vocab_dict, \
                process_mergedict_to_txt, process_mergedict_to_txt_count, load_json_data

jieba.load_userdict("/home/zhangwentao/host/project/information-infer-v2/data/tokens_count.txt")


#需要合并三个模块
#第一个模块：自定义字典组合中文全称
#第二个模块：双塔模型筛选中文全称
#第三个模块：大模型选择最优解


#分成两个阶段进行：1.通过标注数据生成字典进行匹配。2.对于没有匹配到的信息从数据库通用字典中进行匹配

#将所有的字母通过jieba以及目前自定义的词典进行切分查看切分结果

#加载LLM
def init_model(device_id):
    model_path = '/home/zhangwentao/host/LLMs/Models/chatglm3-6b'
    tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device=f'cuda:{device_id}')
    model = model.eval()
    return (model, tokenizer)

models_pool = [init_model(device_id) for device_id in range(2, 4)]
#models_pool = [(i,i) for i in range(4)]
models_ids  = [id for id in range(2)]
data_index = -1
LOCK = threading.Lock()

merge_dict_path = '/home/zhangwentao/host/project/information-infer-v2/data/merge_count.json'
merge_count_dict = load_json_data(merge_dict_path)
result_list = []

def signal_model_infer(model, tokenizer, row, merge_count_dict):
    table_name = row.loc['表名'].strip()
    #将table_name进行切分
    words = jieba.cut(table_name, HMM=True)
    words = '/'.join(words)
    row['切分'] = words
    candidate_items = convert_letter_to_chinese(words, merge_count_dict)
    row['候选项'] = copy.copy(candidate_items)
    #对候选项进行分组
    candidate_groups = []
    sep_len = 3
    if len(candidate_items) > 1:
        while len(candidate_groups) != 1:
            candidate_groups.clear()
            for i in range(0, len(candidate_items), sep_len):
                if i+sep_len <= len(candidate_items):
                    input_str = '[' + ','.join(candidate_items[i:i+sep_len]) + ']'
                elif i == len(candidate_items) - 1:
                    candidate_groups.append(candidate_items[i])
                    break
                else:
                    input_str = '[' + ','.join(candidate_items[i:]) + ']'
                input_str = select_only_one_tamplete.format(input_str)
                response, _ = model.chat(tokenizer, input_str, history=[])
                #print(response)
                #print("++++++++++++++++++")
                try:
                    pattern = r"候选项：'([^']+)'"
                    match = re.search(pattern, response)
                    match_text = match.group(1)
                    candidate_groups.append(match_text)
                except:
                    candidate_groups.clear()
                    candidate_groups.append('模型输出出错')
                    break
            candidate_items = copy.copy(candidate_groups)
    else:
        candidate_groups.append(candidate_items[0])

    row['最佳候选项'] = candidate_groups[0]
    #释放模型推理过程中占用的显存，并将该模型对应的id放到模型队列中，以便没有处理的输入元素可以选择输入
    return row

def model_infer(row, total):
    #print("执行")
    global models_ids, data_index
    #加锁LOCK从全局变量中获取一个model_id
    index = -1
    #print("执行")
    while index == -1:    
        with LOCK:
            if len(models_ids)>0:
                model_id = models_ids.pop(0)
                data_index += 1
                index = data_index
                print(f"total: {total}, now: {index}")
            else:
                continue
        time.sleep(2)
    model, tokenizer = models_pool[model_id]
    #随机间隔一定时间代表数据处理阶段
    result = signal_model_infer(model, tokenizer, row, merge_count_dict)
    #print(f'编号为：{index} 的数据完成处理')
    with LOCK:
        #该模型已经使用完毕，将该模型id重新添加到资源列表中
        models_ids.append(model_id)
    return result

def multiple_infer(original_path, save_folder):
    data_df = pd.read_excel(original_path)
    result_df = pd.DataFrame(columns=['前缀', '次前缀', '表名', '中文标注', '切分', '最佳候选项', '候选项'])
    data_df_list = list(data_df.iterrows())
    total = len(data_df_list)
    #print(type(data_df_list[0][1]))

    #创建4个线程，让这些线程都处于启动状态
    pool_size = 2
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        # 提交任务给线程池
        #executor.map(model_infer, data_df_list)
        futures = [executor.submit(model_infer, row, total) for _, row in data_df_list]

        for future in futures:
            result = future.result()
            result_df.loc[len(result_df)] = result
    result_df.to_csv(os.path.join(save_folder, 'result_candidate_v3.csv'), index=False)

#加载excel文件，对"表名"字段进行切分处理，然后再进行中文全称翻译
def process_all_data(original_path, merge_count_dict, save_folder):
    model, tokenizer = init_model(0)

    data_df = pd.read_excel(original_path)
    result_df = pd.DataFrame(columns=['前缀', '次前缀', '表名', '中文标注', '切分', '最佳候选项', '候选项'])
    for index, row in tqdm(data_df.iterrows(), desc='Processing', leave=True):
        #print(index)
        table_name = row.loc['表名'].strip()
        #将table_name进行切分
        words = jieba.cut(table_name, HMM=True)
        words = '/'.join(words)
        row['切分'] = words
        candidate_items = convert_letter_to_chinese(words, merge_count_dict)
        row['候选项'] = copy.copy(candidate_items)
        #对候选项进行分组
        candidate_groups = []
        sep_len = 3
        if len(candidate_items) > 1:
            while len(candidate_groups) != 1:
                candidate_groups.clear()
                for i in range(0, len(candidate_items), sep_len):
                    if i+sep_len <= len(candidate_items):
                        input_str = '[' + ','.join(candidate_items[i:i+sep_len]) + ']'
                    elif i == len(candidate_items) - 1:
                        candidate_groups.append(candidate_items[i])
                        break
                    else:
                        input_str = '[' + ','.join(candidate_items[i:]) + ']'
                    input_str = select_only_one_tamplete.format(input_str)
                    response, _ = model.chat(tokenizer, input_str, history=[])
                    #print(response)
                    #print("++++++++++++++++++")
                    try:
                        pattern = r"候选项：'([^']+)'"
                        match = re.search(pattern, response)
                        match_text = match.group(1)
                        candidate_groups.append(match_text)
                    except:
                        candidate_groups.clear()
                        candidate_groups.append('模型输出出错')
                        break
                candidate_items = copy.copy(candidate_groups)
        else:
            candidate_groups.append(candidate_items[0])

        row['最佳候选项'] = candidate_groups[0]
        result_df.loc[len(result_df)] = row
    result_df.to_csv(os.path.join(save_folder, 'result_candidate_v2.csv'), index=False)

#对于切分好的首字母简写短语进行中文转换
def convert_letter_to_chinese(split_sentence_letter, merge_count_dict):
    split_sentence_letter = split_sentence_letter.split('/')
    
    split_chinese_list = []
    all_result         = []
    for index, split_letter in enumerate(split_sentence_letter):
        #print("=================", index)
        if split_letter == '_':
            split_chinese_list.append(["_"])
            continue
        #从当前字典中进行查找，首先从第一梯度数据中进行查找，再从第二梯度中进行查找
        if split_letter in merge_count_dict:
            #如果当前首字母简写对应的token频数相差太大则值采取频数高的，如果频数相差不大则都考虑在内
            first_step_max_token = max(merge_count_dict[split_letter][0], key=merge_count_dict[split_letter][0].get) \
                                if len(merge_count_dict[split_letter][0]) > 0 else ""
            sed_step_max_token   = max(merge_count_dict[split_letter][1], key=merge_count_dict[split_letter][1].get) \
                                if len(merge_count_dict[split_letter][1]) > 0 else ""
            
            #先遍历第一梯度的词典
            if len(first_step_max_token) > 0:
                split_chinese_list.append([first_step_max_token])
                first_step_dict = dict(sorted(merge_count_dict[split_letter][0].items(), key=lambda item: item[1], reverse=True))
                for token in first_step_dict:
                    if token == first_step_max_token:
                        continue
                    if first_step_dict[first_step_max_token] / first_step_dict[token] < 5:
                        split_chinese_list[index].append(token)
            
            #遍历第二梯度的词典
            if len(sed_step_max_token) > 0:
                sed_step_dict = dict(sorted(merge_count_dict[split_letter][1].items(), key=lambda item: item[1], reverse=True))
                #print("2梯度数据:",sed_step_dict)
                sign_token = 0
                if first_step_max_token == '':
                    split_chinese_list.append([sed_step_max_token])
                    sign_token = 1
                elif sed_step_max_token not in split_chinese_list[index] and first_step_dict[first_step_max_token] / sed_step_dict[sed_step_max_token] < 5:
                    split_chinese_list[index].append(sed_step_max_token)
                    if sign_token == 0 and sed_step_dict[sed_step_max_token] > first_step_dict[first_step_max_token]:
                        sign_token = 1
                    sign_num = first_step_dict[first_step_max_token] if sign_token==0 else sed_step_dict[sed_step_max_token]
                    for token in sed_step_dict:
                        #print("token:", token)
                        if token == sed_step_max_token:
                            continue
                        if sign_num / sed_step_dict[token] < 5 and token not in split_chinese_list[index]:
                            split_chinese_list[index].append(token)
        else:
            split_chinese_list.append([split_letter])
            #print("split_list:", split_chinese_list[index])
    try:
        assert len(split_sentence_letter) == len(split_chinese_list)
        for index in range(len(split_sentence_letter)):
            temp_list = []
            for token in split_chinese_list[index]:
                if len(all_result) == 0:
                    temp_list.append(token)
                else:
                    for pre_sentence in all_result:
                        #print("+++++++++++", pre_sentence)
                        temp_list.append(pre_sentence+token)
            all_result.clear()
            all_result = temp_list
    except AssertionError as e:
        print("出错：", split_sentence_letter)
    return all_result

def split_sample_name(original_csv_path, save_folder):
    data_df = pd.read_csv(original_csv_path)
    
    result_df = pd.DataFrame(columns=['前缀', '次前缀', '表名', '中文标注', '切分'])
    for index, row in data_df.iterrows():
        words = jieba.cut(row.loc['表名'].strip(), HMM=True)
        words = '/'.join(words)
        row['切分'] = words
        result_df.loc[len(result_df)] = row
    result_df.to_csv(os.path.join(save_folder, 'first_step_count.csv'), index=False)

#数据预处理部分
def data_process(original_file_path, save_folder):
    original_df = pd.read_excel(original_file_path)
    #获取读取出来的两列数据
    #print(original_df.columns.to_list())
    #将前缀、次前缀、表名、中文标注抽取出来
    data_df   = original_df[['前缀', '次前缀', '表名', '中文标注']]
    result_df = pd.DataFrame(columns=['前缀', '次前缀', '表名', '中文标注'])

    for index, line in data_df.iterrows():
        chinese_lable = str(line.loc['中文标注']).strip()
        pinyin_sample = str(line.loc['表名']).strip()
        #print(chinese_lable, '    ', pinyin_sample)
        if len(chinese_lable) == len(pinyin_sample) and not any(char.isupper() for char in chinese_lable):
            result_df.loc[len(result_df)] = line
    result_df.to_csv(os.path.join(save_folder, 'all_data.csv'), index=False)
    #print(len(result_df))
    #print(index)

def infer(example_path):
    #加载示例标签数据，从中随机抽取样例
    all_data_df = pd.read_csv(example_path)
    example_df  = all_data_df.sample(n=5)
    example     = ''
    for i, (index, row) in enumerate(example_df.iterrows()):
        chinese_lable = str(row.loc['中文标注']) 
        #pinyin_quan_lable = ' '.join(lazy_pinyin(chinese_lable))
        #print(pinyin_quan_lable)
        temp_example  = example_tamplete.format(str(row.loc['表名']), chinese_lable)
        #print(temp_example)
        example += temp_example
    
    model, tokenizer = get_model()
    #随便选择一些简写进行输入
    test_example_df = all_data_df.sample(n=10)
    for index, row in test_example_df.iterrows():
        test_table_name = str(row.loc['表名'])
        true_chinese_label = str(row.loc['中文标注'])
        input_contents = instruction_examples.format(example, test_table_name)
        response, _ = model.chat(tokenizer, input_contents, history=[])
        print(input_contents)
        #print('表名：', test_table_name)
        #print('真实标签:', true_chinese_label, '\n', '模型回复：', response)
        #print('+++++++++++++++')

def main():
    excel_path  = '/home/zhangwentao/host/project/information-infer/data/拼音对照.xlsx'
    save_folder = '/home/zhangwentao/host/project/information-infer-v2/data'
    #data_process(excel_path, save_folder)
    #extract_data(excel_path, save_folder)
    #spilt_data(excel_path, save_folder)

    #sql_store_path = '/home/zhangwentao/host/project/information-infer/data/script存储过程.txt'
    #split_sql_store(sql_store_path, save_folder)

    json_prefix_path = '/home/zhangwentao/host/project/information-infer/data/prefix.json'
    acc_cut_path     = '/home/zhangwentao/host/project/information-infer/data/acc_cut.json'
    merge_folder     = '/home/zhangwentao/host/project/information-infer/data'
    #merge_vocab_dict(json_prefix_path, acc_cut_path, merge_folder)

    merge_dict_path  = '/home/zhangwentao/host/project/information-infer-v2/data/merge_count.json'
    #process_mergedict_to_txt(merge_dict_path, save_folder)
    #process_mergedict_to_txt_count(merge_dict_path, save_folder)

    original_csv_path = '/home/zhangwentao/host/project/information-infer/data/all_data.csv'
    #split_sample_name(original_csv_path, save_folder)

    #加载合并之后的涵盖频数的json文件
    #merge_count_dict = load_json_data(merge_dict_path)

    #chinese_name = convert_letter_to_chinese('BQ/_/LS/YZ/DY/JL/K', merge_count_dict)
    #print(chinese_name)

    original_excel_path = '/home/zhangwentao/host/project/information-infer-v2/data/拼音对照.xlsx'
    #process_all_data(original_excel_path, merge_count_dict, save_folder)

    #candidates_list = "['收费_全班结账字典', '收费_全班煎煮字典', '收费_全班结帐字典', '收费_全班记帐字典']"    
    #select_suitable_items(candidates_list)

    multiple_infer(original_excel_path, save_folder)
    #infer(example_path='/home/zhangwentao/host/project/information-infer/data/all_data.csv')

if __name__ == '__main__':
    main()
    
    