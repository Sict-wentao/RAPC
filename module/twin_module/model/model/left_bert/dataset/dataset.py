from torch.utils.data import Dataset
import tqdm
import torch
import random
import jieba
from pypinyin import lazy_pinyin, Style 


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True, left=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_folder_path = corpus_path
        self.encoding = encoding

        self.left = left

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        #print("-------------------------------------------------")
        t1, t2, is_next_label = self.random_sent(item)
        #print("t1: ", t1)
        #print("t2: ", t2)
        #print(t1, t2, is_next_label)
        #对两个token序列进行指定训练策略进行增强
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        #需要一个标签来区分第一部分序列长度和第二部分序列长度
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]

        #将两个tokens列表进行拼接，再将拼接之后的列表按照seq_len进行切分，超出seq_len的元素就删掉
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        #如果当前拼接到一起的输入token列表长度小于指定的最大序列长度，则填充[pad]token
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        #这里需要注意，对于segment_label也需要将pad填充的列表添加到后面，保证bert_input, ber_label,segment_lable的长度都为self.seq_len
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        #print("bert_input: ", bert_input)
        #print("bert_label: ", bert_label)
        #print("segment_label: ", segment_label)

        bert_input = [bert_input]
        bert_label = [bert_label]
        segment_label = [segment_label]
        is_next_label = [is_next_label]

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        #print('---------------------------------------------------')
        #print('')

        return {key: torch.tensor(value) for key, value in output.items()}
    
    #定义获取批数据函数
    def get_batch_items(self, batch_items):
        #遍历每个item，将item中的一样的key进行拼接
        data = {}
        for item in batch_items:
            #print('is_next shape: ', item['is_next'].shape)
            if 'bert_input' not in data.keys():
                data['bert_input']    = item['bert_input']
                data['bert_label']    = item['bert_label']
                data['segment_label'] = item['segment_label']
                data['is_next']       = item['is_next']
                continue
            #print('样例: ', item)
            data['bert_input']    = torch.cat((data['bert_input'], item['bert_input']), 0)
            data['bert_label']    = torch.cat((data['bert_label'], item['bert_label']), 0)
            data['segment_label'] = torch.cat((data['segment_label'], item['segment_label']), 0)
            data['is_next']       = torch.cat((data['is_next'], item['is_next']), 0)

        #print("batch size bert_input shape: ", data['bert_input'].shape)
        return data

    def random_word(self, sentence):
        #先采用jieba进行分词
        cn_tokens = list(sentence)
        #cn_tokens = jieba.cut(sentence, cut_all=False)
        
        #判断当前是不是首字母简称预处理模型，如果是则需要将每个token转换成拼音首字母简写
        if self.left:
            en_tokens = [''.join(lazy_pinyin(word, style=Style.FIRST_LETTER)).upper() for word in cn_tokens]
            tokens = en_tokens
        else:
            tokens = cn_tokens

        #先对切分之后的文本进行一次初次清洗，将其中的标点符号等不是字母的都给剔除掉
        #tokens = [token for token in tokens if token[0].isalpha()]

        #print("拼音切分样本：", tokens)
        #tokens = sentence.split()
        #output_label中存放的是对切分之后的token进行一定概率随机处理操作后，新的tokens列表中token对应的词典中的索引
        output_label = []

        #对token列表中的每一个token进行随机操作，将85%的token保持不变，15%的token发生指定改变
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                
                #将确认要发生变化的token中80%的变成mask
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                #将确认要发生变化的token中10%的随机变成词典中的任意一个token
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                #将确认要发生变化的token中10%的保持原token不变
                # 10% randomly change token to current token
                else:
                    #保持该token不变
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                #将未处理之前的token的索引添加到output_lable中
                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                #保持该token不变，如果该token不存在词典中，则将返回unk_index
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                #如果该token不是要进行变化的那一部分，则给他打标签为0并添加到output_lable中
                output_label.append(0)
        #output_label中有85%的索引为0，15%的索引为未处理前的token的原始index

        return tokens, output_label

    #这里返回两个文本t1、t2，以及标签0/1，
    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]

#写一个批处理函数
