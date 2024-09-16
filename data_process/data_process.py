import os
import sys
import gzip
import jsonlines
import jieba
import re
import random
import json
import pandas as pd
from collections import Counter

root = os.getcwd()
root = '/'.join(root.split('/')[:-1])
print(root)
sys.path.append(root)

from module.data_module.DataModule import DataModule

#（1）读取每一个json文件，并将json文件中长度较长的中文部分的内0容抽取出来
def extrac_content_jsonl_to_txt(jsonl_file_folder, save_folder=''):
    #获取解压文件的后缀名
    jsonl_file_name_list = os.listdir(jsonl_file_folder)
    
    for file in jsonl_file_name_list:
        jsonl_path = os.path.join(jsonl_file_folder, file)
        name = file.split('.')[0]
        output_path = os.path.join(save_folder, name+'.txt')
        with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                for line in jsonlines.Reader(jsonl_file):
                    #line_num += 1
                    contents = []
                    for row in line['段落']:
                        #只将内容长度大于30的文本保留，短文本剔除
                        if len(row['内容']) > 40:
                            contents.append(row['内容'])
                    if len(contents) > 0:
                        contents_str = '\n'.join(contents)
                        output_file.write(contents_str)

#这里就需要采用jieba对当前长文本进行切分，然后再统计语料中中文token的拼音首字母对应的词库并统计成自定义词典
#首先需要对当前文本进行进一步处理，通过(, 。 ;)进行切分，每一行只能保留两个部分，统计按照符号切分之后的文本字段中中文的比例，如果中文的比例大于90%则保留，否则剔除
def process_txt(txt_folder, save_folder):
    original_txt_list = os.listdir(txt_folder)

    output_path = os.path.join(save_folder, 'vocab.txt')
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for txt_file in original_txt_list:
            origianl_txt_path = os.path.join(txt_folder, txt_file)
            name = txt_file.split('.')[0]
            #output_path = os.path.join(save_folder, name+'.txt')
            with open(origianl_txt_path, 'r', encoding='utf-8') as original_txt_file:
                for line in original_txt_file.readlines():
                    contents_pire = []
                    #对原始文本中的数据进行清洗
                    #根据指定字符进行切分
                    line_list = re.split('，|。|；', line)
                    #将空字符串直接删掉
                    line_list = [span for span in line_list if len(span)>0 and len(span.replace(' ','').replace('\t','').replace('\r','').replace('\v','').replace('\f','').replace('\n','')) > 0]
                    #line_list = line.split('，。；')
                    #print(line_list)
                    #经过分隔符切分之后可以分成两部分的文本留下来，否则直接跳过改行
                    if len(line_list) > 1:
                        #对切分之后的每个中文字段求取其相对应的中文比例
                        pattern = re.compile(r'[\u4e00-\u9fa5]')
                        line_cn_scale_list = [round(len(re.findall(pattern, span))/len(span), 2) for span in line_list]
                        #按照相邻的中文字段拼接成一对，再判断这两个里面中文字符的比例是否都大于某个阈值，如果大于某个阈值则将这一对文本保留，否则就舍弃掉
                        for index in range(1, len(line_list)):
                            temp_content = []
                            if line_cn_scale_list[index-1] > 0.9 and line_cn_scale_list[index] > 0.9 and len(line_list[index-1]) > 5 and len(line_list[index]) > 5:
                                temp_content.append(line_list[index-1])
                                temp_content.append(line_list[index])
                                contents_pire.append(temp_content)
                    #如果改行抽取出来多个文本pire，则将这些文本对分别写入到输出文件中
                    if len(contents_pire)>0 :
                        #print(contents_pire)
                        contents_pire = ['\t'.join(pire) for pire in contents_pire]
                        contents_str = '\n'.join(contents_pire)
                        if contents_str[-1] != '\n':
                            output_file.write(contents_str + '\n')

# 对当前样本数据进行筛选，按照8：2的比例进行切分，保存成train.txt和dev.txt文件
def split_data(vocab_path, target_folder, min_token_num=0):
    train_path = os.path.join(target_folder, f'train-v2.txt')
    dev_path   = os.path.join(target_folder, f'dev-v2.txt')
    #target_txt_path = os.path.join(target_folder, f'vocab_{line_num}.txt')
    num = 0
    with open(vocab_path, 'r', encoding='utf-8') as file:
        with open(train_path, 'w', encoding='utf-8') as train_file , open(dev_path, 'w', encoding='utf-8') as dev_file:
            for line in file.readlines():
                line_list = list(line)
                if len(line_list) < min_token_num:
                    continue
                pro = random.random()
                if pro>0.8:
                    # 保存到dev.txt文件
                    dev_file.write(line)
                else:
                    # 保存到train.txt文件
                    train_file.write(line)

# 对当前数据集抽取line_num条数据
def extract_line_num(vocab_path, target_folder, line_num):
    train_path = os.path.join(target_folder, f'train_{line_num}.txt')
    num = 0
    with open(vocab_path, 'r', encoding='utf-8') as file:
        with open(train_path, 'w', encoding='utf-8') as train_file:
            for line in file.readlines():
                train_file.write(line)
                num +=1
                if num == line_num:
                    break

#从所有的样本数据中进一步提取数据，将token长度大于指定阈值min_token_num的样本抽取出来，line_num表示抽取指定行数的样本
def extract_test_txt(vocab_path, target_folder, line_num=0, min_token_num=25):
    # 抽取文件路径
    target_txt_path = os.path.join(target_folder, f'vocab_{line_num}.txt')
    #读取指定路径的语料文件
    num = 0
    with open(vocab_path, 'r', encoding='utf-8') as file:
        with open(target_txt_path, 'w', encoding='utf-8') as tr_file:
            for line in file.readlines():
                #只保留长文本，短文本直接删除，保证token不能少于一个固定阈值
                if line_num != 0 and num > line_num:
                    break
                line_list   = line.split('\t')
                line_tokens = []
                line_tokens  += jieba.cut(line_list[0], cut_all=False)
                line_tokens  += jieba.cut(line_list[1], cut_all=False)
                if len(line_tokens) < min_token_num:
                    continue
                num += 1
                tr_file.write(line)

#对预训练语料进行处理
def data_process_TwinTower(vocab_path, target_folder):
    target_txt_path = os.path.join(target_folder, f'TwinTower.txt')
    #num = 10
    with open(vocab_path, 'r', encoding='utf-8') as file:
        last_str = ''
        with open(target_txt_path, 'w', encoding='utf-8') as tar_file:
            for line in file.readlines():
                line_list = line.split('\t')
                #将这两个数据分别写入到目标文件中
                if last_str[0:-1] == line_list[0]:
                    tar_file.write(line_list[1])
                else:
                    tar_file.write(line_list[0] + '\n')
                    tar_file.write(line_list[1])
                last_str = line_list[1]
                #num -= 1
                #if num == 0:
                #    break


#抽取部分数据作为样本数据
def data_extract(vocab_path, target_folder, line_nums=1000):
    target_txt_path = os.path.join(target_folder, f"train-{line_nums}.txt")
    with open(vocab_path, 'r', encoding='utf-8') as file:
        with open(target_txt_path, 'w', encoding='utf-8') as tar_file:
            for line in file.readlines():
                tar_file.write(line)
                line_nums -= 1
                if line_nums == 0:
                    break

# 对文件进行清洗, 关键词提取
def clean_sql_store(data_folder, target_path, threshold=10):
    # 加载停用词表
    stop_word_path = os.path.join(data_folder, 'stop_word', 'stop_word.txt')
    stop_word = {}
    with open(stop_word_path, 'r') as stop_file:
        for line in stop_file.readlines():
            line = line.strip('\n')
            stop_word[line] = 0

    # 对当前sql中文数据进行分词，将长度大于二并且不是停用词的都给添加进来
    sql_store_path = os.path.join(data_folder, 'MedicalTwinTower', 'all_data', 'sql_store.txt')
    sql_content = []
    with open(sql_store_path, 'r') as sql_file:
        for line in sql_file.readlines():
            # 对改行进行分词操作
            line_list = list(jieba.cut(line, cut_all=True))
            # 通过停用词进行统计
            sql_content += [word for word in line_list if word not in stop_word and len(word)>0 and word!=' ']
    
    # 对标注数据进行分词统计
    table_train_path = os.path.join(data_folder, 'MedicalTwinTower', 'all_data', 'train.xlsx')
    table_train_data = pd.read_excel(table_train_path)
    train_word_list = []
    for index, row in table_train_data.iterrows():
        line = row['中文标注']
        line = line.replace('_', ' ')
        line_list = list(jieba.cut(line, cut_all=True))
        #print(line_list)
        #break
        train_word_list += [word for word in line_list if word!=' ' and len(word)>0]
        #print(train_word_list)
        #break

    # 对外部数据进行读取
    table_outer_path = os.path.join(data_folder, 'MedicalTwinTower', 'all_data', 'table.xlsx')
    table_outer_data = pd.read_excel(table_outer_path)
    outer_word_list = []
    for index, row in table_outer_data.iterrows():
        line = row['说明']
        line_list = list(jieba.cut(str(line), cut_all=True))
        outer_word_list += [word for word in line_list if word not in stop_word and len(word)>0 and word!=' ']

    # 将当前统计的所有词合并成成一个词频字典，和通过关键词挖掘出来的进行对比，将二者进行融合
    # 读取通过特征提取抽取出来的词
    clean_json_path = os.path.join(data_folder, 'MedicalTwinTower', 'clean_data', 'clean_words.txt')
    clean_word_dict = {}
    with open(clean_json_path, 'r') as file:
        for line in file.readlines():
            line = line.replace('\n', '')
            line = line.replace('\'', '\"')
            line_json = json.loads(line)
            clean_word_dict[line_json['word']] = 1

    all_word_tokens = sql_content + train_word_list + outer_word_list
    all_word_dict   = Counter(all_word_tokens)
    sorted_counter = sorted(all_word_dict.items(), key=lambda x: x[1], reverse=True)

    target_path = os.path.join(data_folder, 'MedicalTwinTower', 'key_word', 'keyWord.txt')
    target_word_list = []
    with open(target_path, 'w') as file:
        for item in sorted_counter:
            if len(item[0])==1 and item[1]>threshold:
                file.write(item[0] + '\n')
            elif len(item[0])>1 and item[0] not in clean_word_dict and item[1] > 5:
                #target_word_list.append(item[0])
                file.write(item[0] + '\n')

        # 将所有的特征抽取得到的关键词写入
        for item in clean_word_dict:
            file.write(item + '\n')

# 通过BGE模型语义相似度计算，生成难负采样数据
# 读取指定的文件，加载其中的字符串表名以及相对应的中文标注名，生成相对应的难负样本
def generate_train_data(original_path, target_path, sample_num=4):
    cfg = {}
    #cfg['field_data_path'] = '/home/zhangwentao/host/project/paper/data/vocab/sql_store.txt'
    #cfg['key_word_path']   = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/key_word/keyWord.txt'
    cfg['letter_map_cn_path'] = '/home/zhangwentao/host/project/paper-v1.3/data/vocab'
    cfg['custom_letter_path'] = '/home/zhangwentao/host/project/paper-v1.3/data/vocab'
    cfg['top_k'] = 8
    cfg['batch_size'] = 20480
    cfg['max_length'] = 512

    cfg['embedding_model'] = '/home/zhangwentao/host/project/paper-v1.3/module/twin_module/model/model/right_bert/bge-large-zh-v1.5'

    cfg['custom_jieba']  = True
    dataTool = DataModule(cfg)

    original_df = pd.read_excel(original_path)
    target_path = os.path.join(target_path, 'last_train.txt')
    original_data = []
    # 读取excel文件中的表名列和中文标注列
    with open(target_path, 'w', buffering=1) as file:
        for index, row in original_df.iterrows():
            if index < 467:
                continue 
            else:
                if index > 934:
                    break
                print(f'第 {index} 行')
                #temp_list = [row['表名'], row['中文标注']]
                similarity_cn_score = dataTool.select_top_k_embedding(letter_name=row['表名'],
                                                                    label_cn=row['中文标注'],
                                                                    k=sample_num*8)
                #print(len(similarity_cn_score))
                # 将这个正负样本数据写入到txt文件中
                #temp_list = []
                offset = 0
                end_index = sample_num*2
                if end_index>len(similarity_cn_score):
                    end_index = len(similarity_cn_score)

                # 存放正样本
                positive_cn_list = []
                # 先添加正样本
                data_num = end_index//2
                for i in range(data_num):
                    left_data = row['表名']
                    if i==0:
                        right_data = row['中文标注']
                        offset = -1
                    else:
                        # 在这里一直往后找，对于相同的直接跳过
                        right_data = similarity_cn_score[i+offset][0]
                        while right_data in positive_cn_list:
                            offset += 1
                            right_data = similarity_cn_score[i+offset][0]
                    positive_cn_list.append(right_data)
                    label = 1
                    #print("写入")
                    file.write(left_data + '\t' + right_data + '\t' + str(label) + '\n')
                
                # 添加负样本数据
                # 存放负样本
                negative_cn_list = []
                offset = 0
                score_len = len(similarity_cn_score)
                for i in range(score_len-1, score_len-1-data_num, -1):
                    right_data = similarity_cn_score[i+offset][0]
                    while right_data in negative_cn_list:
                        offset -= 1
                        right_data = similarity_cn_score[i+offset][0]
                    negative_cn_list.append(right_data)
                    label = 0
                    #print("写入")
                    file.write(left_data + '\t' + right_data + '\t' + str(label) + '\n')

# 对每一个首字母简写进行中文映射，并且将对应的中文和

def main():
    data_folder = '/home/zhangwentao/host/project/paper-v1.3/data'
    target_path = ''
    clean_sql_store(data_folder, target_path, threshold=100)
        
if __name__ == '__main__':
    """gzip_file_folder = '/home/zhangwentao/host/project/paper/data/pretrain_data/wiki_jsonl'
    save_folder      = '/home/zhangwentao/host/project/paper/data/pretrain_data/wiki'
    extrac_content_jsonl_to_txt(gzip_file_folder, save_folder)"""

    #txt_folder = '/home/zhangwentao/host/project/paper/data/pretrain_data/wiki'
    #save_folder = '/home/zhangwentao/host/project/paper/data/pretrain_data/wiki_pire'
    #process_txt(txt_folder, save_folder)

    #test_str = """2012年9月2日，弗洛伦齐首次代表罗马队首发出场，并在比赛中射进个人在罗马的首球。帮助罗马客场3比1击败国际米兰。之后他占据球队的主力位置，2012/13赛季为罗马在联赛出场36次，射进3球。"""
    #str_list = re.split('，|。|；', test_str)
    #print(str_list)

    #读取合并到一起的vocab.txt文件，并查看其中的数据是否是正确的
    """vocab_txt = '/home/zhangwentao/host/project/paper/data/pretrain_data/wiki_pire/vocab.txt'
    with open(vocab_txt, 'r', encoding='utf-8') as f:
        #print(len(f))
        for line in f:
            print(line)
            break"""
    

    #抽取部分数据做实验
    vocab_path    = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/train_data/v2/train.txt'
    target_folder = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/train_data/v2'
    #extract_test_txt(vocab_path, target_folder, min_token_num=50)
    split_data(vocab_path, target_folder, min_token_num=10)
    #extract_line_num(vocab_path, target_folder, line_num=500)

    #data_process_TwinTower(vocab_path, target_folder)

    #vocab_path    = '/home/zhangwentao/host/project/paper/data/TwinTower/original_train.txt'
    #target_folder = '/home/zhangwentao/host/project/paper/data/TwinTower'
    #data_extract(vocab_path, target_folder, line_nums=500)

    #main()


    #----------------------------
    """original_path = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/all_data/train.xlsx'
    target_path = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/train_data'
    sample_num  = 4
    generate_train_data(original_path, target_path, sample_num)"""