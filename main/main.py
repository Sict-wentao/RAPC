import os
import pandas as pd
import json

import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
# print('主函数： ', root_path)
sys.path.append(root_path)

from module.data_module.DataModule import DataModule
from module.twin_module.TwinModule import TwinModule
from module.llm_module.LLMModule import LLMModule
from module.twin_module.model.dataset.vocab import WordVocab
from config.cfg import A_config
from util.util import evaluate_bleu

class AllModule:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # 初始化一个DataModule
        self.dataModule = DataModule(self.cfg)
        if self.cfg['cot_infer'] :
            self.llmModule = LLMModule(self.cfg)
        else:
            self.twinModule = TwinModule(self.cfg)
        #self.llmModule  = LLMModule(self.cfg)
    
    def data_process(self, letter_name):
        # 对一个表名进行切分，并进行中文组合，返回所有的结果
        letter_name, cn_all_results = self.dataModule.get_letter_cn_list(letter_name, threashold=0.5, top_k=self.cfg['top_k_split'])
        cn_all_results = [''.join(split_list) for split_list in cn_all_results]
        inputs = {}
        inputs['letter_name']    = letter_name
        inputs['cn_all_results'] = cn_all_results
        return inputs
    
    def twin_process(self, inputs, k):
        # 通过双塔模型将概率值最大的k个候选项筛选出来
        sort_output, sort_index = self.twinModule.predict(inputs)
        
        # 将返回概率值最大的k个候选项筛选出来
        riddle_results = []
        for index in sort_index[:k]:
            riddle_results.append(inputs['cn_all_results'][index])

        return riddle_results

    def llm_process(self, riddle_results, result_k):
        best_result = self.llmModule.signal_model_infer(riddle_results, result_k)
        return best_result
    
    def convert_letter_to_cn(self, letter_name, label_cn=None, riddle_k=100, result_k=10):
        inputs = self.data_process(letter_name)
        riddle_results = self.twin_process(inputs, riddle_k)
        # 评估双塔模型筛选出来的候选项的bleu score
        if label_cn is not None:
            bleu_socre = evaluate_bleu(label_cn, riddle_results, top_k=50)
        
        #best_result = self.llm_process(riddle_results, result_k)
        return bleu_socre
    
    # 读取一个文件，获取中文中双塔筛选排名前100的元素
    def retrun_top_k_cn(self, letter_name, top_k=100, result_save_path=None):
        inputs = self.data_process(letter_name)
        riddle_results = self.twin_process(inputs, top_k)
        # 将这个结果存储起来
        return riddle_results
    
    def cot_llm(self, letter_name):
        # 获取当前缩写字符串所对应的中文词典信息
        all_keys = self.dataModule.get_all_dict(letter_name)
        # 调用LLM进行生成
        cns = self.llmModule.cot_infer_signal(letter_name, all_keys)
        return cns

# 读取当前文件，将其中的首字母中文映射字符串填充
def map_cn_letter_name(model, source_file_path, target_file_path):
    # 通过panda 读取原始excel文件
    excel_df = pd.read_excel(source_file_path)
    top_k = 2000
    #result_json = {}
    # 将双塔模型筛选结果保存到一个jsonl文件中
    with open(target_file_path, 'w', encoding='utf-8') as tfile:
        for index, row in excel_df.iterrows():
            if index < 486 or (index > 533 and index <557):
                continue
            print(f'第{index}行')
            temp_json = {}
            letter_name  = row['表名']
            inputs       = model.data_process(letter_name)
            #print('获取到相应的中文候选项了')
            top_k_result = model.twin_process(inputs, top_k)
            # 将结果写入到json
            temp_json['letter_name'] = letter_name
            temp_json['cn_name']     = top_k_result
            tfile.write(json.dumps(temp_json, ensure_ascii=False)+'\n')

# 读取一个jsonl文件，获取每个字典元素中的top_k个中文候选项
def get_topk_jsonl(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            item = json.loads(line)

# 读取一个jsonl文件，获取其中所有的letter_name,并调用cot直接推理出n个中文候选项
def cot_cns_jsonl(allModule, jsonl_path):
    target_path = os.path.dirname(jsonl_path)
    target_path = os.path.join(target_path, 'cot_infer.jsonl')
    print()
    num = 0
    with open(jsonl_path, 'r', encoding='utf-8') as sfile, open(target_path, 'w', encoding='utf-8') as tfile:
        for line in sfile:
            if num < 257:
                num += 1
                continue
            print(num)
            data = json.loads(line)
            letter_name = data['letter_name']
            cns = allModule.cot_llm(letter_name)
            # 将当前结果写入到jsonl文件
            temp_json = {}
            temp_json['letter_name'] = letter_name
            temp_json['cn_name']     = cns
            tfile.write(json.dumps(temp_json, ensure_ascii=False)+'\n')
            num += 1

# 按行读取每一行中文候选项，将其分组进行筛选（随机进行分组， 按照k个一组进行分组处理，挑选出每一组中效果最好的候选项）

if __name__ == '__main__':
    cfg = A_config
    allModule = AllModule(cfg)
    #letter_name = 'SF_YS_ZZSQK'
    #label_cn    = '收费_医生_转诊申请库'
    """letter_name = 'BQ_YS_BASSK'
    label_cn    = '病区_医生_病案手术库'
    bleu_score = allModule.convert_letter_to_cn(letter_name, label_cn=label_cn, riddle_k=1000)
    print('最终结果为：', bleu_score)"""
    #print('候选项中最可能的结果为：', best_result)

    """# 调用文件，生成检索控制器的结果
    source_file = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/all_data/test.xlsx'
    target_file = '/home/zhangwentao/host/project/paper-v1.3/data/experiment/v3/large/test-large.jsonl'
    map_cn_letter_name(allModule, source_file, target_file)"""

    """letter_name = 'KH_SSFZK'
    allModule.cot_llm(letter_name)"""

    jsonl_path = '/home/zhangwentao/host/project/paper-v1.3/data/experiment/v3/small/test-target.jsonl'
    cot_cns_jsonl(allModule, jsonl_path)
