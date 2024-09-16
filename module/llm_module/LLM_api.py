from openai import OpenAI
import os
import re
import json

Instruction = """指令: 你是一名专业的医疗领域的数据库工程师，你需要对数据库中的表进行命名，需要保证命名结果简洁并且具有一定的可解释性，接下来会给你提供一些表名候选项，你需要从候选项中选择一个出来你觉得最具有一定说明性以及可解释性的候选项，并给出原因，一步一步分析，回答格式要严格按照先回答哪个选项，再回答具体原因的格式进行回答。

输入: A.'考核_HIS项目收据调入设置'  B.'考核_HIS项目数据调入设置'   

输出: 答案为B。因为B选项具有可解释性： 每个关键词都很清晰地描述了表的内容。"考核"表明这是与考核相关的数据，"HIS项目"指明这是与医院信息系统项目相关的数据，而"数据调入设置"则说明这是与数据导入设置有关的表。B选项具有层次结构： 命名中的词语似乎具有一定的层次结构，先是考核，然后是HIS项目，最后是数据调入设置。这种结构有助于组织和理解数据库中的不同表。

"""
model_name = 'qwen2-72b-instruct'

class LLM_api:
    def __init__(self):
        self.api_key    = 'sk-75635c9662804784be28b7874b35113c'
        self.base_url   = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        #self.model_name = 'qwen2-72b-instruct'
        self.model_name = 'qwen2-57b-a14b-instruct'

        self.temperature= 0.8
        self.top_p      = 0.8

        self.client     = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def get_response(self, option_a, option_b):
        user_input = f"输入:A.{option_a} B.{option_b}"
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': Instruction},
                {'role': 'user', 'content': user_input}],
            temperature=self.temperature,
            top_p=self.top_p
            )
        response = completion.model_dump_json()
        response = json.loads(response)
        #print(response)
        #print(type(response))
        return response['choices'][0]['message']['content']
    def extract(self, response):
        try:
            pattern = r"答案为[A-Z]"
            match = re.search(pattern, response)
            match_text      = match.group(0)[-1]
            index_candidate = ord(match_text) - ord('A')
            return index_candidate # 0/1
        except:
            return 3
        
if __name__ == '__main__':
    model = LLM_api()
    option_a, option_b = '考核_HIS项目收据调入设置', '考核_HIS项目数据调入设置'
    response = model.get_response(option_a, option_b)
    state = model.extract(response)
    print(response)
    print(state)