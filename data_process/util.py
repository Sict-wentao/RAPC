import os
import pandas as pd
import json
import jieba
import re
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

#对输入文本进行抽取
def extract_line_cn_and_en(content):

    chinese_pattern = re.compile('[\u4e00-\u9fa5]')  # 匹配中文字符的正则表达式

    # 检查文本中是否包含中文
    if chinese_pattern.search(content) is None:
        return None  # 如果没有中文，直接返回None或者空列表，表示跳过
    
    mixed_pattern = re.compile('[\u4e00-\u9fa5]+')
    mixed_text = mixed_pattern.findall(content)
    return mixed_text

#将数据库存储文件中的中文文本信息抽取出来
def extract_sql_store_cn(sql_store_path, save_folder):
    with open(sql_store_path, 'r') as f:
        contents = f.readlines()

    output_txt = os.path.join(save_folder, 'sql_store.txt')
    with open(output_txt, 'w') as file: 
        for line in tqdm(contents, total=len(contents), leave=True, miniters=100):
            line_list = extract_line_cn_and_en(line)
            if line_list is None or len(line_list) == 0:
                continue
            content = ' '.join(line_list)
            if len(content) > 20:
                file.write(content+'\n')

        
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

# 读取一个字典映射表，将所有的首字母都提取出来
def generate_custom_letter_by_map_dict(map_cn_path, target_path):
    #letter_map_cn_path = os.path.join(self.cfg['letter_map_cn_path'], 'letter_cn.json')
    with open(map_cn_path, 'r') as f:
        etter_map_cn = json.load(f)
    # 将所有的key保存至custom.txt文件
    custom_path = os.path.join(target_path, 'custom.txt')
    with open(custom_path, 'w') as f:
        for key in etter_map_cn.keys():
            f.write(key + '\n')

# 读取jsonl文件，将每一行的字段数据都写入到csv文件
def conver_jsonl_to_csv(jsonl_path):
    csv_folder = jsonl_path.split('/')[:-1]
    csv_path   = '/'.join(csv_folder) + '/target.csv'
    data_df = pd.DataFrame({'letter_name': [''], 'top_10': ['']})
    with open(jsonl_path, 'r') as json_file:
        for line in json_file:
            data = json.loads(line)
            data_df.loc[len(data_df)] = [data['letter_name'], data['top_10_result']]
    # save df
    data_df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    #json_file_path = '/home/zhangwentao/host/project/information-infer/data/acc_cut.json'
    #check_en_key(json_file_path)

    #content = """取消结算、退费、红冲时间 因需要传如单个时间可以进行查询，需为大于等于"""
    #result = extract_line_cn_and_en(content)
    #print(result)

    """sql_store_path = '/home/zhangwentao/host/project/paper/data/script存储过程.txt'
    save_folder_path = '/home/zhangwentao/host/project/paper/data'
    extract_sql_store_cn(sql_store_path, save_folder_path)"""

    """map_cn_path = '/home/zhangwentao/host/project/paper-v1.3/data/vocab/letter_cn.json'
    target_path = '/home/zhangwentao/host/project/paper-v1.3/data/vocab'
    generate_custom_letter_by_map_dict(map_cn_path, target_path)"""

    jsonl_path = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/all_data/target-3327.jsonl'
    conver_jsonl_to_csv(jsonl_path)