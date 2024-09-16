import os
import jieba

# 关键词抽取
# 通过tf-idf算法从当前切分之后的词典中找出
def dict_extract(vocab_path):
    ## 文本预处理
    with open(vocab_path, 'r') as file:
        lines = file.readlines()
        # 处理每一行数据
        for line in lines:
            token_list = []
            # 分词
            line_list = line.split('\t')
            token_list += jieba.cut(line_list[0], cut_all=True)
            token_list += jieba.cut(line_list[1], cut_all=True)
    # 去除停用词
    
    ## 关键词提取
    # 词典生成及idf计算

# 新词发现实现专有词典构建
# 新词发现主要有两个模块：
#     （1）离线模块：从海量文本语料中挖掘出高质量的短语及其属性
#     （2）在线模块：识别给定文本中出现的短语供下游模块使用
# 统计出现频次
# 求点间互信息
# 左右熵

# 功能：PMI过滤噪声词


# word2vec拓展关键词库