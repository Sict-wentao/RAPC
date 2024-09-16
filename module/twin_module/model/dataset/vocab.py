import pickle
import tqdm
import jieba
import os
from pypinyin import lazy_pinyin, Style
from collections import Counter


class TorchVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
        """
        
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        #max_size就是词汇表中所有词汇的个数以及添加进来特殊的一些标识符的个数之和
        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        #按照词汇表中从小到大的出现频率进行遍历，如果出现的频率小于设定的min_freq或者当前已经确定下来的词表中的个数已经达到了最大的个数要求
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        #stoi中存放的是确定下来的词汇表中的每个词汇对应的索引
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    #可用于判断两个TorchVocab是否相等
    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    #返回词汇表中词汇的个数
    def __len__(self):
        return len(self.itos)

    #更新词汇表中每个词汇对应的索引值
    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}


    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        self.counter = counter
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_folder):
        vocab_path = os.path.join(vocab_folder, 'vocab.pkl')

        #保存该分词模型到pkl文件
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)
        #保存自定义词典到txt文件,对counter进行排序
        counter = self.counter.copy()
        sorted(counter)
        dict_counter = dict(counter)
        vocab_txt_path = os.path.join(vocab_folder, 'vocab.txt')
        with open(vocab_txt_path, 'w') as f:
            for k in dict_counter:
                f.write(k + ' ' + str(dict_counter[k]) + '\n')    
        

# Building Vocab with text files
# 在此处对词表操作进行更新，生成新的词表文件
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        #统计folder_path路径下的所有的词料txt文件
        print("Building Vocab")
        #先分别读取所有的txt文件，然后再对文件中的所有的文本数据进行结巴分词，然后再将分词结果转换成拼音首字母，再统计成词表
        counter = Counter()
        for line in tqdm.tqdm(texts):
            words = []
            #切分每一行的文本数据
            if isinstance(line, list):
                words = line
            else:
                #采用jieba进行分词操作
                line = line.replace('\n', '')
                line_list = line.split('\t')
                #print(line_list)
                words += jieba.cut(line_list[0], cut_all=False)
                words += jieba.cut(line_list[1], cut_all=False)
                words = [''.join(lazy_pinyin(word, style=Style.FIRST_LETTER)).upper() for word in words]
            for word in words:
                counter[word] += 1
        print('计数器生成结束')

        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    #将一个本文序列转换成每个词汇在词汇表中的索引值，形成相应的索引序列
    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            #通过jieba对中文文本进行分词
            sentence = jieba.cut(sentence, cut_all=False)
            sentence = [''.join(lazy_pinyin(token, style=Style.FIRST_LETTER)).upper() for token in sentence]
            #sentence = sentence.split()
        #遍历句子序列中每个词汇在词汇表中对应的索引值，如果没有在加载后的词汇表中找到，则将其按照自定义的unk_index进行处理
        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        #根据句子不同的属性进行不同的划分，这里猜想：如果是要进行生成的句子，则结尾需要添加<sos>,如果是判断生成结束的则要在句子尾部添加<eos>
        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        #这里需要给句子填充，保证每一个文本序列都是按照统一的长度进行处理的
        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq
    
    #将一个索引序列转换成相对应的文本序列
    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

def build():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()

    #vocab = WordVocab(args.corpus_path, max_size=args.vocab_size, min_freq=args.min_freq)

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.output_path)

if __name__ == '__main__':
    build()
    #test_str = 'abu'
    #print(test_str.upper())
