from collections import Counter
import numpy as np
import re,os
import glob
import six
import codecs
import math
import threading
import time
import json
import jieba


# 停用词加载
def get_stop_word(stop_word_path):
    # 停用词列表，默认使用哈工大停用词表
    f = open(stop_word_path, encoding='utf-8')
    stop_words = []
    for stop_word in f.readlines():
        stop_words.append(stop_word[:-1])
    
    return stop_words

# 语料生成器
def text_generator(file_path):
    # 读取文件
    with open(file_path, encoding='utf-8') as file:
        for line in file.readlines():
            # 返回这一行文本数据
            line = line.replace('\n', '')
            yield line
    #d = codecs.open(file_path, encoding='utf-8').read()

# jieba分词，统计分词结果，对于长度大于1的进行统计
def tokenizer_tool(file_path, all_dicts, target_path):
    """
        input:
            file_path : 原始语料库文件
            all_dicts:  新词发现检索出来的所有的词
    """
    # 获取所有的all_dicts['word']，将其转换成一个list
    all_dicts_words = []
    for new_word in all_dicts.keys():
        all_dicts_words.append(new_word['word'])

    #jieba_dicts = []
    with open(file_path, 'r') as original_file, open(target_path, 'w') as target_file:
        lines = original_file.readlines()
        for line in lines:
            tokens = jieba.cut(line, cut_all=True)
            for token in tokens:
                if len(token) > 1 and token not in all_dicts_words:
                    # 将结果写入到目标文件中
                    target_file.write(token + '\n')

# 通过设定指定阈值对前面全部新词进行筛选
def select_token(file_path, max_p=1000, min_entropy=1, max_score=100, min_score=100):
    target_file = os.path.dirname(file_path)
    target_file = os.path.join(target_file, 'clean_wrods.txt')

    # 读取存放新词词典的txt文件
    with open(file_path, 'r', encoding='utf-8') as file, open(target_file, 'w') as target_file:
        for line in file.readlines():
            line = line.replace('\n', '')
            line = line.replace('\'', '\"')
            new_word = json.loads(str(line))
            # 对该词进行阈值筛选
            if new_word['pmi']<max_p and min(new_word['left_entropy'], new_word['right_entropy']) > min_entropy:
                target_file.write(str(new_word)+'\n')

# 新词发现类定义
class NewWordFind():
    def __init__(self, n_gram=5, max_p=2, min_entropy=1, max_score=100, min_score=2):
        """
            input:
                n_gram:int    n_gram的粒度
                max_p: int    最大点互间信息指数
                min_entropy: int    左右熵阈值
                max_score: int      综合得分最大阈值
                min_score: int      综合得分最小阈值
        """
        self.n_gram = n_gram
        self.max_p  = max_p
        self.min_entropy = min_entropy
        self.max_score = max_score
        self.min_score = min_score
    
    # 将text进行n_gram
    def n_gram_words(self, text):
        """
            input: 
                text: string  
            return:
                words_freq: Dict   词频 字典
        """
        words = []
        for i in range(1, self.n_gram+1):
            for j in range(len(text)-i+1):
                words.append(text[j:j+i])
        
        # 统计词频
        words_freq = dict(Counter(words))
        #print('词频：', words_freq)
        new_words_freq = {}
        for word, freq in words_freq.items():
            new_words_freq[word]=freq
        
        return new_words_freq

    # 功能：PMI过滤噪声词,互信息统计，计算文本片段凝聚度
    def PMI_filter(self, word_freq_dic):
        """
            describe:
                通过对当前词的内部凝聚度进行统计，PMI计算公式为 pmi = log2 [p(x, y) / (p(x)p(y))], 按照原始PMI计算公式来看，当pmi绝对值越接近0说明凝聚度越高
                PMI改进算法： pmi = (p(x)p(y)) / p(x, y) ，按照这个公式则值越大越好，所以要设置一个上限pmi来进行凝聚度筛选。
            input:
                words_freq: Dict     词频 字典
            return:
                new_words_dic: Dict   过滤噪声后，剩余的新词
        """
        new_wrods_dic = {}
        for word in word_freq_dic:
            if len(word) == 1:
                pass
            else:
                p_x_y = min([word_freq_dic.get(word[:i])*word_freq_dic.get(word[i:]) for i in range(1, len(word))])
                mpi   = p_x_y/word_freq_dic.get(word)

                if mpi < self.max_p:
                    new_wrods_dic[word] = [mpi]

        return new_wrods_dic
    
    # 计算字符列表的熵
    def calculate_entropy(self, char_list):
        """
            input:
                char_list: list     字符列表
            return:
                entropy:  float     熵
        """
        char_freq_dic = dict(Counter(char_list))
        entropy = (-1)*sum([char_freq_dic.get(char)/len(char_list)*np.log2(char_freq_dic.get(char)/len(char_list)) for char in char_freq_dic])
        
        return entropy
    
    # 通过熵阈值限定词字典中过滤出最终的新词
    def Entropy_left_right_filter(self, condinate_wrods_dic, text, target_path):
        """
            describe:
                用于计算该词的自由度，首先统计在当前语料中，该词紧邻左右字符的集合，然后分别计算左熵和右熵。
                在这里，如果熵值越大则说明该词所含信息越丰富，可能是一个新词的概率越大，因此需要给一个下限熵，用于筛选新词。
            input:
                condinate_words_dic: Dict     限定词字典
                text: String       句子
            return:
                final_words_list: List       最终的新词列表
        """
        final_words_list = []
        condinate_words_list = list(condinate_wrods_dic.keys())
        # 直接读取文件
        file = open(target_path, 'w')
        # 共享索引，用于遍历词典序列
        index = 0
        index_lock = threading.Lock()
        final_lock = threading.Lock()

        def worker():
            nonlocal  index
            while True:
                with index_lock:
                    if index >= len(condinate_words_list):
                        break
                    temp_index = index
                    index += 1
                word = condinate_words_list[temp_index]
                left_right_char = re.findall('(.)%s(.)'%word, text)

                left_char = [i[0] for i in left_right_char]
                left_entropy = self.calculate_entropy(left_char)

                right_char = [i[1] for i in left_right_char]
                right_entropy = self.calculate_entropy(right_char)
                score = condinate_wrods_dic[word][0] - min(left_entropy, right_entropy)
                if min(right_entropy, left_entropy) > self.min_entropy and score < self.max_score and score > self.min_score:
                    temp_dict = {
                            "word": word,
                            "pmi": condinate_wrods_dic[word][0],
                            'left_entropy':left_entropy,
                            "right_entropy":right_entropy,
                            "score": score
                        }
                    # 因为所有线程要对同一list进行添加操作，因此需要加锁
                    with final_lock:
                        # 直接写入到文件中
                        file.write(str(temp_dict) + '\n')
                        final_words_list.append(temp_dict)

        # 多线程处理数据
        threads = []
        num_threads = 1000
        for i in range(num_threads):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        #等待所有线程结束
        for thread in threads:
            thread.join()

        """for word in condinate_wrods_dic.keys():
            left_right_char = re.findall('(.)%s(.)'%word, text)
            
            left_char = [i[0] for i in left_right_char]
            #print('左字符: ', left_char)
            left_entropy = self.calculate_entropy(left_char)

            right_char = [i[1] for i in left_right_char]
            #print('右字符: ', right_char)
            right_entropy = self.calculate_entropy(right_char)
            score = condinate_wrods_dic[word][0] - min(left_entropy, right_entropy)
            if min(right_entropy, left_entropy) > self.min_entropy and score < self.max_score and score > self.min_score:
                temp_dict = {
                        "word": word,
                        "pmi": condinate_wrods_dic[word][0],
                        'left_entropy':left_entropy,
                        "right_entropy":right_entropy,
                        "score": score
                    }
                final_words_list.append(temp_dict)
                print(temp_dict)"""

        final_words_list = sorted(final_words_list, key=lambda x: x['score'], reverse=True)
        return final_words_list
    
    def save_wrods_dic(self, words_list, save_path):
        """
            input:
                words_list: list, [{"word": ,"pmi": ,'left_entropy':,"right_entropy":,"score": }]
                save_path: str, 保存路径
        """
        # 获取当前时间
        now_time = int(time.time())
        timeArray = time.localtime(now_time)
        noe_Time = time.strftime("%m%d%H%M%S", timeArray)
        save_path = os.path.join(save_path, f'sql_key_dict_{noe_Time}.txt')
        with open(save_path, 'w') as file:
            for word in words_list:
                file.write(str(word)+'\n')

def main():
    stop_word_path = '/home/zhangwentao/host/project/paper/data/stop_word/stop_word.txt'
    file_path      = '/home/zhangwentao/host/project/paper/data/vocab/sql_store.txt'
    stop_word = get_stop_word(stop_word_path)

    # 初始化新词发现类
    # 初始化一些参数阈值配置
    n_gram = 4
    min_p  = 1
    min_entropy = 0.5
    max_score = 1000
    min_score = 0

    newWordFind = NewWordFind(n_gram=n_gram, min_p=min_p, min_entropy=min_entropy, max_score=max_score, min_score=min_score)

    text = codecs.open(file_path, encoding='utf-8').read()
    text = text.replace(u'\u3000', ' ').strip()
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    # print(text[:100])

    # 处理停用词
    for sword in stop_word:
        text.replace(sword, '')

    print('词频统计...')
    words_freq = newWordFind.n_gram_words(text)
    print('凝聚度统计筛选...')
    new_words_dic  = newWordFind.PMI_filter(words_freq)
    print("自由度筛选...")
    new_words_list = newWordFind.Entropy_left_right_filter(new_words_dic, text, target_path='/home/zhangwentao/host/project/paper/data/vocab/all_words_0531.txt')

    print("保存结果到txt文件")
    #newWordFind.save_wrods_dic(new_words_list, save_path='/home/zhangwentao/host/project/paper/data/vocab')

    """print("new words print ...") 
    for new_words in new_words_list:
        print(f'{new_words}')"""

# 基于现有的新词词典以及分词词库进行专业词典的生辰
def final_words():
    max_p = 400
    min_entropy = 1
    file_path = '/home/zhangwentao/host/project/paper/data/vocab/all_words_0531.txt'
    select_token(file_path, max_p=max_p, min_entropy=min_entropy)
    

if __name__ == '__main__':
    # 构建语料生成器
    #main()
    final_words()