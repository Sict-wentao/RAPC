from torch.utils.data import Dataset
import tqdm
import torch
import random
import jieba
from pypinyin import lazy_pinyin, Style

class TwinTowerDataset(Dataset):
    def __init__(self, 
                 corpus_path, 
                 left_vocab,
                 right_vocab, 
                 seq_len, 
                 encoding='utf-8',
                 corpus_lines=None,
                 on_memory=True,
                 pretrain=True):
        
        self.left_vocab   = left_vocab
        self.right_vocab  = right_vocab 
        self.seq_len      = seq_len

        self.on_memory    = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path  = corpus_path
        self.encoding     = encoding
        self.pretrain     = pretrain

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        #训练语料数据没有加载到内存中，每次加载一部分到内存
        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, index):
        # 通过index获取到当前数据列表中指定索引的中文文本
        # left_t是一个首字母组成的token list，right_t是一个中文token组成的list
        left_t, right_t, label = self.random_sent(index)

        left_input_ids     = left_t[:self.seq_len]
        left_segment_label = [1 for _ in range(len(left_input_ids))]
        left_padding       = [self.left_vocab.pad_index for _ in range(self.seq_len-len(left_input_ids))]

        left_input_ids.extend(left_padding), left_segment_label.extend(left_padding)
        left_model_input = {"input_ids": torch.tensor([left_input_ids]),
                            "segment_label": torch.tensor([left_segment_label])}
        right_model_input = {
            "input_ids": right_t['input_ids'],
            "token_type_ids": right_t['token_type_ids'],
            "attention_mask": right_t['attention_mask']
        }
        
        output = {"left_input" :  left_model_input,
                  "right_input": right_model_input,
                  "label"      : torch.tensor([label], dtype=torch.long)}
        
        return output
    
    def get_batch_items(self, batch_items):
        data = {}
        for item in batch_items:
            #print('is_next shape: ', item['is_next'].shape)
            if len(data.keys()) == 0:
                data['left_input']    = item['left_input']
                data['right_input'] = item['right_input']
                data['label']       = item['label']
                continue
            data['left_input']['input_ids']       = torch.cat((data['left_input']['input_ids'], item['left_input']['input_ids']), 0)
            data['left_input']['segment_label']   = torch.cat((data['left_input']['segment_label'], item['left_input']['segment_label']), 0)

            data['right_input']['input_ids']      = torch.cat((data['right_input']['input_ids'], item['right_input']['input_ids']), 0)
            data['right_input']['token_type_ids'] = torch.cat((data['right_input']['token_type_ids'], item['right_input']['token_type_ids']), 0)
            data['right_input']['attention_mask'] = torch.cat((data['right_input']['attention_mask'], item['right_input']['attention_mask']), 0)
            
            data['label'] = torch.cat((data['label'], item['label']), 0)

        #print("batch size bert_input shape: ", data['bert_input'].shape)
        return data


    # 预训练阶段调用的数据构造函数
    def random_sent(self, index):
        if self.pretrain:
            cn_sentence = self.get_corpus_line(index)
            #默认需要返回的标签是正类别
            label = 1
            #按照概率进行采样
            if random.random() > 0.5:
                #通过jeiba对当前样本数据进行分词
                #精确查询，按理来说应该是添加自己的自定义词典，现在先按照jieba自己的词典进行切分，后续再进行优化
                #left_token_list  = jieba.cut(cn_sentence, cut_all=False)
                left_token_list  = list(cn_sentence)
                #将当前切分之后的中文token转换成字母首字母简写
                left_token_list  = [''.join(lazy_pinyin(word, style=Style.FIRST_LETTER)).upper() for word in left_token_list]
                #需要转换成token_id_list
                left_token_list  = [self.left_vocab.stoi.get(token, self.left_vocab.unk_index) for token in left_token_list]
                #用right_model的tokenizer处理该文本,直接就是token_id_list
                right_token_list = self.right_vocab(cn_sentence, max_length=self.seq_len, padding='max_length', truncation=True, return_tensors='pt')
            else:
                label = 0
                #进行负采样，随机再给right_model获取一个中文文本，或者对该文本中的首字母序列进行随机转换形成一个负样本（这样子就保证了有效长度不变）
                if random.random() > 0.8:
                    #从训练数据中再随机抽取一个文本数据作为right_model的输入数据
                    #left_token_list   = jieba.cut(cn_sentence, cut_all=False)
                    left_token_list = list(cn_sentence)
                    left_token_list   = [''.join(lazy_pinyin(word, style=Style.FIRST_LETTER)).upper() for word in left_token_list]
                    left_token_list  = [self.left_vocab.stoi.get(token, self.left_vocab.unk_index) for token in left_token_list]
                    right_cn_sentence = self.get_random_line()
                    right_token_list  = self.right_vocab(right_cn_sentence, max_length=self.seq_len, padding='max_length', truncation=True, return_tensors='pt')
                else:
                    #随机替换left_token_list中的token
                    #left_token_list  = jieba.cut(cn_sentence, cut_all=False)
                    left_token_list  = list(cn_sentence)
                    left_token_list  = [''.join(lazy_pinyin(word, style=Style.FIRST_LETTER)).upper() for word in left_token_list]
                    right_token_list = self.right_vocab(cn_sentence, max_length=self.seq_len, padding='max_length', truncation=True, return_tensors='pt')
                    for i,token in enumerate(left_token_list):
                        if random.random() < 0.1:
                            left_token_list[i] = random.randint(5, len(self.left_vocab)-1)
                        else:
                            left_token_list[i] = self.left_vocab.stoi.get(token, self.left_vocab.unk_index) 
            left_token_list = [self.left_vocab.sos_index] + left_token_list + [self.left_vocab.eos_index]   
            return left_token_list, right_token_list, label
        else:
            line = self.get_corpus_line(index).strip('\n').split('\t')
            #print(len(line))
            letter_name, right_cn, label = line[0], line[1], int(line[2])
            left_token_list = list(letter_name)
            left_token_list  = [self.left_vocab.stoi.get(token, self.left_vocab.unk_index) for token in left_token_list]
            right_token_list = self.right_vocab(right_cn, max_length=self.seq_len, padding='max_length', truncation=True, return_tensors='pt')
            return left_token_list, right_token_list, label

            
    
    def get_corpus_line(self, index):
        if self.on_memory:
            return self.lines[index]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, 'r', encoding=self.encoding)
                line = self.file.__next__()
            return line
    
    def get_random_line(self):
        #语料数据全都加载到内存中
        if self.on_memory:
            #获取负样本
            random_index = random.randint(0, len(self.lines)-1)
            return self.lines[random_index]
        
        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line
