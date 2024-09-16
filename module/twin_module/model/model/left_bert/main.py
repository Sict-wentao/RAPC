import argparse

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("/home/zhangwentao/host/project/paper/model/model/left_bert")

from model import BERT
from trainer import BERTTrainer
from dataset import BERTDataset, WordVocab


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=786, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=12, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=12, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument('-start', "--start_epoch", type=int, default=0, help="first epoch")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument('-st', "--save_step", type=int, default=None, help='每一个epoch中训练次数达到save_step就保存一次')
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--left", type=bool, default=True, help="默认是预训练左边字母简写预训练模型")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
    parser.add_argument("--model_weight", type=str, default=None, help="可以添加模型权重文件，以初始化模型权重")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0001, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))
    #print("词表tokens：", vocab.itos)

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory,
                                left=args.left)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory, left=args.left) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=train_dataset.get_batch_items)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=test_dataset.get_batch_items) \
        if test_dataset is not None else None

    #print(train_dataset.__getitem__(1))
    print("Building BERT model")
    start_epoch = args.start_epoch
    optim, optim_schedule = None, None
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
    if args.model_weight :
        print("Loading BERT model weight: ", args.model_weight)
        state = torch.load(args.model_weight)
        start_epoch      = state['epoch'] + 1
        model_state_dict = state['model_state_dict']
        optim            = state['optimizer']
        optim_schedule   = state['optimizer_schedule']
        # 加载模型参数
        bert.load_state_dict(model_state_dict)
    # print("测试epoch: ", bert['epoch'])

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, 
                          optim=optim, optim_schedule=optim_schedule)

    print("Training Start")
    for epoch in range(start_epoch, args.epochs):
        trainer.train(epoch, save_step=args.save_step, output=args.output_path)
        if epoch%2 == 0:
            trainer.save(epoch, weight_folder_path=args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch, output=args.output_path)
        
        #保存最后一次训练权重为last.pt
        if epoch+1 == args.epochs:
            trainer.save(epoch, is_last=True, weight_folder_path=args.output_path)
    
if __name__ == '__main__':
    train()
