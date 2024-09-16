import os
import json

def clean_jsonl(jsonl_path):
    # 读取jsonl文件，获取其中的所有的字典
    clean_jsonl_path = os.path.dirname(jsonl_path)
    clean_jsonl_path = os.path.join(clean_jsonl_path, 'clean.jsonl')
    keys = []
    with open(jsonl_path, 'r') as sfile, open(clean_jsonl_path, 'w') as tfile:
        for line in sfile:
            temp_data = json.loads(line)
            if temp_data['letter_name'] not in keys:
                # 将结果写入到当前文件中
                tfile.write(json.dumps(temp_data, ensure_ascii=False)+'\n')
                keys.append(temp_data['letter_name'])

def check_jsonl(source_1, source_2):
    s1, s2 = [], []
    with open(source_1, 'r') as s1_file, open(source_2, 'r') as s2_file:
        for line in s1_file:
            temp_data = json.loads(line)
            s1.append(temp_data['letter_name'])
        for line in s2_file:
            temp_data = json.loads(line)
            s2.append(temp_data['letter_name'])

        print('在1中的，2中没有的')
        for letter_name in s1:
            if letter_name not in s2:
                print(letter_name)
        
        print('2有，1没有的')
        for letter_name in s2:
            if letter_name not in s1:
                print(letter_name)



if __name__=='__main__':
    #jsonl_path = '/home/zhangwentao/host/project/paper-v1.3/data/experiment/v3/large/test-large.jsonl'
    #clean_jsonl(jsonl_path)

    source_1, source_2 = '/home/zhangwentao/host/project/paper-v1.3/data/experiment/v3/base/clean.jsonl', '/home/zhangwentao/host/project/paper-v1.3/data/experiment/v3/large/clean.jsonl'
    check_jsonl(source_1, source_2)