import re
import copy
import random
import json
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel

import sys 
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-3])
#print('模块：', root_path)
sys.path.append(root_path)
from config.prompt import select_only_one_tamplete, elo_tamplete, cot_generate_candidates
from module.llm_module.LLM_api import LLM_api

class LLMModule:
    def __init__(self, cfg):
        # 利用transformer库加载LLM
        """self.model_path = cfg['LLM_Model_Path']
        self.device = cfg['LLM_device']
        self.tokenizer, self.model = self.init_model(self.model_path)"""
        self.cfg = cfg

        # 本地部署
        if self.cfg["local_infer"]:
            # 利用vllm加载LLM进行推理
            if self.cfg['n_return'] != 0:
                self.sampling_params = SamplingParams(n=100,
                                                    temperature=0.9,
                                                    top_p=0.8,
                                                    max_tokens=500)
            else:
                self.sampling_params = SamplingParams(temperature=0.8,
                                                    top_p=0.95,
                                                    max_tokens=50)
            self.model = LLM(model=cfg['LLM_Model_Path'],
                            tensor_parallel_size=4,
                            enable_prefix_caching=True,
                            trust_remote_code=True)
        else:
            self.model = LLM_api()
        
    def init_model(self, model_path):
        model_path = self.model_path
        tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device=self.device)
        model = model.eval()
        return tokenizer, model
    
    def signal_model_infer_old(self, candidate_items, k=10):
        #对候选项进行分组
        candidate_groups = []
        sep_len = 3
        if len(candidate_items) > 1:
            while len(candidate_groups) != 1:
                candidate_groups.clear()
                for i in range(0, len(candidate_items), sep_len):
                    temp_items = []
                    if i+sep_len <= len(candidate_items):
                        input_str = '[' + ','.join(candidate_items[i:i+sep_len]) + ']'
                        temp_items = candidate_items[i:i+sep_len]
                    elif i == len(candidate_items) - 1:
                        candidate_groups.append(candidate_items[i])
                        break
                    else:
                        input_str = '[' + ','.join(candidate_items[i:]) + ']'
                        temp_items = candidate_items[i:]
                    input_str = select_only_one_tamplete.format(input_str)
                    response, _ = self.model.chat(self.tokenizer, input_str, history=[])
                    #print(response)
                    #print("++++++++++++++++++")
                    try:
                        pattern = r"最佳候选项为：'([^']+)'"
                        match = re.search(pattern, response)
                        match_text = match.group(1)
                        print('挑选结果：', match_text)
                        candidate_groups.append(match_text)
                    except:
                        #candidate_groups.clear()
                        candidate_groups.extend(temp_items)
                        #candidate_groups.append('模型输出出错')
                        #break
                candidate_items = copy.copy(candidate_groups)
                if len(candidate_items) <= k:
                    return candidate_items
        else:
            candidate_groups.append(candidate_items[0])

        return candidate_groups
    
    def signal_model_infer(self, candidate_items, k=10):
        #对候选项进行分组
        sort_item = ['A', 'B', 'C', 'D', 'E', 'F']
        candidate_groups = []
        sep_len = 3
        if len(candidate_items) > 1:
            while len(candidate_groups) != 1:
                # 打乱候选列表
                random.shuffle(candidate_items)
                candidate_groups.clear()
                inputs_list = []
                items_list  = []
                # 对当前候选项中的选项分组，每一组最多分成sep_len个
                for i in range(0, len(candidate_items), sep_len):
                    temp_items = []
                    input_str  = []
                    if i+sep_len <= len(candidate_items):
                        end_index = i+sep_len
                        # 这一组可以分成sep_len个候选
                        #input_str = '[' + ','.join(candidate_items[i:i+sep_len]) + ']'
                        temp_items = candidate_items[i:i+sep_len]
                        items_list.append(temp_items)
                    elif i == len(candidate_items) - 1:
                        candidate_groups.append(candidate_items[i])
                        break
                    else:
                        end_index = len(candidate_items)
                        #input_str = '[' + ','.join(candidate_items[i:]) + ']'
                        temp_items = candidate_items[i:]
                        items_list.append(temp_items)
                    for index in range(i, end_index):
                        input_str.append(sort_item[index-i] + '.' + candidate_items[index])
                    input_str = '\t'.join(input_str)
                    #print('拼接字符串: ', input_str)
                    input_str = select_only_one_tamplete.format(input_str)
                    inputs_list.append(input_str)
                    #print('input prompt: ', input_str)
                    # ---------------------原始采用transformer库进行推理-----------------
                    """response, _ = self.model.chat(self.tokenizer, input_str, history=[])
                    #print(response)
                    #print("++++++++++++++++++")
                    try:
                        pattern = r"答案为[A-Z]"
                        match = re.search(pattern, response)
                        match_text      = match.group(0)
                        index_candidate = ord(match_text) - ord('A')
                        result_candidate= candidate_items[i+index_candidate]
                        print('选项：', match_text)
                        print('挑选结果：', result_candidate)
                        # 将该答案对应的候选项映射为字符串
                        candidate_groups.append(result_candidate)
                    except:
                        #candidate_groups.clear()
                        candidate_groups.extend(temp_items)
                        #candidate_groups.append('模型输出出错')
                        #break"""
                    # -----------------------------------------------------------------
                # 使用vllm进行并发加速推理
                outputs = self.model.generate(inputs_list, 
                                             sampling_params=self.sampling_params)
                # 对vllm输出进行解析
                for i, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
                    try:
                        pattern = r"答案为[A-Z]"
                        match = re.search(pattern, generated_text)
                        match_text      = match.group(0)[-1]
                        index_candidate = ord(match_text) - ord('A')
                        result_candidate= candidate_items[i*sep_len+index_candidate]
                        #print('选项：', match_text)
                        #print('挑选结果：', result_candidate)
                        # 将该答案对应的候选项映射为字符串
                        candidate_groups.append(result_candidate)
                    except:
                        #candidate_groups.clear()
                        candidate_groups.extend(items_list[i])
                        #candidate_groups.append('模型输出出错')
                        #break
                candidate_items = copy.copy(candidate_groups)
                if len(candidate_items) <= k:
                    return candidate_items
        else:
            candidate_groups.append(candidate_items[0])

        return candidate_groups

    def update_ratings(self, options, winner, loser, K):
        R_winner = options[winner]
        R_loser = options[loser]

        E_winner = 1 / (1 + 10 ** ((R_loser - R_winner) / 400))
        E_loser = 1 / (1 + 10 ** ((R_winner - R_loser) / 400))

        options[winner] = R_winner + K * (1 - E_winner)
        options[loser] = R_loser + K * (0 - E_loser)

    def extract(self, response):
        try:
            pattern = r"答案为[A-Z]"
            match = re.search(pattern, response)
            match_text      = match.group(0)[-1]
            index_candidate = ord(match_text) - ord('A')
            return index_candidate # 0/1
        except:
            return 3

    def elo_eval_infer(self, source_jsonl, target_jsonl):
        INITIAL_RATING = 1500
        K = 32
        num = 0
        with open(source_jsonl, 'r', encoding='utf-8') as sfile, open(target_jsonl, 'w', encoding='utf-8', buffering=1) as tfile:
            for line in sfile:
                if num < 139:
                    num += 1
                    continue
                temp_json = {}
                line_json = json.loads(line)
                cn_list   = line_json['cn_name'][:100]
                print("line: ", num)
                
                # 调用elo算法计算分数
                options = {f"{i}": INITIAL_RATING for i in range(len(cn_list))}
                if self.cfg['local_infer']:
                    batch_index = []
                    for index_a in range(len(cn_list)-1):
                        for index_b in range(index_a+1,len(cn_list)):
                            batch_index.append((index_a, index_b))
                    for b_index in range(len(batch_index)//self.cfg['batch_size']):
                        sample_list= batch_index[b_index*self.cfg['batch_size']:(b_index+1)*self.cfg['batch_size']]
                        input_list = ['A.'+cn_list[a]+'\t'+'B.'+cn_list[b] for a,b in sample_list]
                        input_list = [elo_tamplete.format(user_input) for user_input in input_list]
                        outputs    = self.model.generate(input_list,
                                                        sampling_params=self.sampling_params)
                        # 统计该批量每一个对比结果，计算分数
                        for i,output in enumerate(outputs):
                            response = output.outputs[0].text
                            index   = self.extract(response)
                            index_a = sample_list[i][0]
                            index_b = sample_list[i][1]
                            if index == 0:
                                winner = index_a
                                loser  = index_b
                            elif index == 1:
                                winner = index_b
                                loser  = index_a
                            else:
                                continue
                            self.update_ratings(options,
                                                str(winner),
                                                str(loser),
                                                K)
                else:
                    for index_a, option_a in enumerate(cn_list):
                        for index_b, option_b in enumerate(cn_list[index_a+1:]):
                            if option_a != option_b:
                                # 输入到LLM，得到获胜的候选项。
                                if self.cfg['local_infer']:
                                    input_str  = 'A.'+option_a+'\t'+'B.'+option_b
                                    user_input = elo_tamplete.format(input_str) + '\n' + '输出:'
                                    response   = self.model.generate(user_input,
                                                                    sampling_params=self.sampling_params)
                                    # generated_text = output.outputs[0].text
                                    # 批量推理
                                else:
                                    response = self.model.get_response(option_a=option_a, option_b=option_b)
                                index = self.model.extract(response)
                                if index == 0:
                                    winner = index_a
                                    loser  = index_a+1+index_b
                                else:
                                    winner = index_a+1+index_b
                                    loser  = index_a
                                self.update_ratings(options,
                                                    str(winner),
                                                    str(loser),
                                                    K)
                # 排序options
                options = sorted(options.items(),
                                 key=lambda x: x[1],
                                 reverse=True)
                result = []
                for item in options:
                    result.append(cn_list[int(item[0])])
                # 保存结果
                temp_json['letter_name'] = line_json['letter_name']
                temp_json['top_result']  = result
                tfile.write(json.dumps(temp_json, ensure_ascii=False)+'\n')
                num += 1

    def cot_infer_signal(self, letter_name, key_cns, max_loop=5):
        # 将已知信息输入到cot模板中
        llm_input = cot_generate_candidates.replace('map_dict', str(key_cns)).replace('letter_name', (letter_name))
        cns = []
        loop = 0
        def generate_cns(input):
            outputs = self.model.generate([llm_input],
                             sampling_params=self.sampling_params)
            for i, output in enumerate(outputs):
                # 通过正则化将候选结果抽取出来
                cns = []
                error_num = 0
                for response in output.outputs:
                    match = re.search(r'中文标签应为：“(.*?)”', response.text)
                    if match:
                        cn = match.group(1)
                        if cn in cns:
                            error_num += 1
                        else:
                            cns.append(cn)
                    else:
                        error_num += 1
                #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #print("error_num: ", error_num)
                #print(cns)
            return error_num, cns
        
            """print("输出结果：", len(output.outputs))
            generated_text = output.outputs[0].text
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(generated_text)"""

        while len(cns) < 100 and loop<max_loop:
            loop += 1
            error_num, temp_cns = generate_cns(llm_input)
            for cn in temp_cns:
                if cn not in cns:
                    cns.append(cn)
        return cns

        
# 读取jsonl文件，获取每一行字典数据，取其中top_k个中文候选项，进行vllm推理
def process_cn_by_llm(llm, source_jsonl, target_jsonl):
    top_k = 10
    with open(source_jsonl, 'r', encoding='utf-8') as sfile, open(target_jsonl, 'w', encoding='utf-8') as tfile:
        for line in sfile:
            temp_json = {}
            line_json = json.loads(line)
            #print(line_json['cn_name'][1])
            # 将每一个样本的200个给到大模型进行筛选top k个
            result = llm.signal_model_infer(line_json['cn_name'][:200], k=top_k)
            # 模型处理结果写入到一个文件中
            temp_json['letter_name']          = line_json['letter_name']
            temp_json[f'top_{top_k}_result']  = result
            tfile.write(json.dumps(temp_json, ensure_ascii=False)+'\n')
            #print("结果： ", result)

if __name__ == '__main__':
    """cfg = {}
    cfg['LLM_Model_Path'] = '/home/zhangwentao/host/LLMs/Models/ChatGLM2/chatglm2-6b'
    cfg['LLM_device']     = 'cuda:1'
    llmModule = LLMModule(cfg)
    candidate_items = ['病区_医技查询大类病区对应库', '病区_一级查询大类病区对应库', '病区_医技查询系统大类病区对应库', '病区_一级查询系统大类病区对应库', '病区_医技查询代理病区对应库', '病区_一级查询代理病区对应库', '病区_医技查询系统代理病区对应库', '病区_一级查询系统代理病区对应库']
    llmModule.signal_model_infer(candidate_items, k=2)
    print(llmModule)"""
    
    cfg = {}
    cfg['LLM_Model_Path'] = '/home/zhangwentao/host/LLMs/Models/chatglm3-6b'
    cfg['LLM_device']     = 'cuda:0'
    cfg['local_infer']    = True
    cfg['batch_size']     = 32
    cfg['n_return']       = 0
    llmModule = LLMModule(cfg) 

    #source_jsonl = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/all_data/his-3327.jsonl'
    #target_jsonl = '/home/zhangwentao/host/project/paper-v1.3/data/MedicalTwinTower/all_data/target-3327.jsonl'
    source_jsonl = '/home/zhangwentao/host/project/paper-v1.3/data/experiment/v3/small/cot_infer.jsonl'
    target_jsonl = '/home/zhangwentao/host/project/paper-v1.3/data/experiment/v3/small/cot-target-GLM3.jsonl'
    # process_cn_by_llm(llm=llmModule, source_jsonl=source_jsonl, target_jsonl=target_jsonl)
    llmModule.elo_eval_infer(source_jsonl, target_jsonl)