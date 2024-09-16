import os
import sys
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")

from model.TwinTower import TwinTower
from utils.utils import init_weight

class TwinTowerTrainer:
    def __init__(self,
                 left_model,
                 right_model,
                 mergeNet_weight_path=None,
                 hidden_size=768,
                 train_dataloader=None,
                 test_dataloader=None,
                 optimizer='adam',
                 loss='crossentropy',
                 metric = ['auc', 'accuracy'],
                 lr: float = 1e-4,
                 betas=(0.9, 0.999),
                 cuda_devices=None,
                 with_cuda = True,
                 log_freq=2,
                 save_step=10):
        
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        #self.device = torch.device("cpu")

        self.left_model  = left_model
        self.right_model = right_model

        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader

        self.step = 0
        self.save_step = save_step

        #定义双塔模型
        self.model = TwinTower(hidden_size=hidden_size, left_model=left_model, right_model=right_model, mergeNet_layer=1)
        #print("原始：", self.model.state_dict()['right_model.encoder.layer.3.output.dense.weight'])

        if mergeNet_weight_path:
            #print('加载权重')
            # 如果目前mergeNet参数权重存在则加载进来
            for name, p in self.model.named_parameters():
                if 'left_model' in name or 'right_model' in name:
                    #print(name)
                    p.requires_grad = False
            state = torch.load(mergeNet_weight_path)
            self.model.mergeNet.load_state_dict(state['mergeNet_state_dict'])
        else:
            #初始化模型参数,将left_model和right_model模型参数更新状态设置为False
            for name, p in self.model.named_parameters():
                if 'left_model' in name or 'right_model' in name:
                    #print(name)
                    p.requires_grad = False
            self.model.mergeNet.apply(init_weight)
        
        #print("初始化：", self.model.state_dict()['right_model.encoder.layer.3.output.dense.weight'])
        
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for Model" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        # 初始化优化器
        """if mergeNet_weight_path:
            self.optim = state['optimizer']
            # 更新优化器超参数
            new_lr = lr
            for param_group in self.optim.param_groups:
                param_group["lr"] = new_lr
        else:"""
            #定义优化器
        self.optim = self._get_optim(filter(lambda p : p.requires_grad, self.model.parameters()), optimizer, lr)
        #self.optim_schedule = ScheduledOptim(self.optim, sel)
        #定义损失函数
        self.loss_func = self._get_loss_func(loss) 
        #保存测试结果
        self.motrics   = {}

        self.log_freq = log_freq

        self.best_acc = 0
        self.step_best_acc = 0
        self.train_acc = 0
        self.dev_acc   = 0
    
    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader, train=True)
    
    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, data_loader, train=True):
        
        str_code  = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        
        avg_loss      = 0
        total_correct = 0
        total_element = 0

        abs_path = os.getcwd()
        output_dir = os.path.dirname(abs_path)
        if str_code=='train':
            log = 'train_log'
            self.model.train()
        else:
            log = 'dev_log'
            self.model.eval()
        log_path = os.path.join(output_dir, 'output', f'{log}.txt')
        model_weight_path = os.path.join(output_dir, 'output', 'weight')
        if not os.path.exists(model_weight_path):
            os.makedirs(model_weight_path)

        for i, data in data_iter:
            #torch.cuda.empty_cache()
            self.step += 1
            input_data = {}
            for key, value in data.items():
                if key not in input_data.keys():
                    input_data[key] = {}
                if type(value)==dict:
                    for sub_key, sub_value in data[key].items():
                        input_data[key][sub_key] = data[key][sub_key].to(self.device)
                else:
                    input_data[key] = data[key].to(self.device)
            #前向传播
            if str_code == 'train':
                output = self.model.forward(input_data["left_input"], input_data["right_input"])
            else:
                with torch.no_grad():
                    output = self.model.forward(input_data["left_input"], input_data["right_input"])
            
            loss = self.loss_func(output, input_data["label"])

            if str_code == 'train':
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            #print("更新后：", self.model.module.state_dict()['right_model.encoder.layer.3.output.dense.weight'])
            
            output  = torch.where(output>0.5, torch.ones_like(output), torch.zeros_like(output))
            output  = torch.argmax(output, dim=1) # 这里返回的是每一个输出最大值的位置索引，为0或1，shape为(batch_size, 1)
            correct = output.eq(input_data['label']).sum().item()
            #correct  = output.argmax(dim=-1).eq(input_data["label"]).sum().item()

            avg_loss += loss.item()
            total_correct += correct
            total_element += input_data["label"].nelement()

            avg_acc = total_correct / total_element * 100
            iter_acc = correct / input_data["label"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_acc": avg_acc,
                "iter_acc": iter_acc,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }
            
            del(input_data)
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                #将当前训练模型的结果保存到指定的文件中
                with open(log_path, 'a', encoding='utf-8') as file:
                    file.write(str(post_fix) + '\n')

            # 当前训练step达到save_step，保存参数
            if self.step % self.save_step == 0:
                self.save(epoch, self.step, is_last=True, folder_path=model_weight_path)
            
            # 保存效果最好的step
            if iter_acc > self.step_best_acc:
                self.step_best_acc = iter_acc
                self.save(epoch, self.step, best=True, folder_path=model_weight_path)
        
        if str_code == 'train':
            self.train_acc = total_correct / total_element * 100
        
        if str_code == 'test':
            self.dev_acc = total_correct / total_element * 100

        #保存在测试集上准确率最高的模型
        if str_code == 'test' and self.dev_acc > self.best_acc:
            self.best_acc = self.dev_acc
            self.save(epoch, best=True, folder_path=model_weight_path)

        self.save(epoch, is_last=True, folder_path=model_weight_path)
        
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)
    
    def save(self, epoch, step=None, best=False, is_last=False, folder_path="output"):
        state          = {}
        state['epoch'] = epoch
        state['step']  = step
        mergeNet_state_dict = self.model.module.mergeNet.cpu().state_dict()
        #mergeNet_state_dict = self.model.mergeNet.cpu().state_dict()
        state['mergeNet_state_dict'] = mergeNet_state_dict
        state['optimizer'] = self.optim

        # 保存效果最好的step模型参数
        if step and best:
            step_best_path = os.path.join(folder_path, 'step_best.pt')
            torch.save(state, step_best_path)
        
        if step and is_last:
            step_last_path = os.path.join(folder_path, 'step_last.pt')
            torch.save(state, step_last_path)

        if not step and best:
            best_path = os.path.join(folder_path, 'test_best.pt')
            torch.save(state, best_path)
        
        if not step and is_last:
            last_path = os.path.join(folder_path, 'epoch_last.pt')
            torch.save(state, last_path)
        
        self.model.module.mergeNet.to(self.device)

    def _get_optim(self, parameters, optimizer, lr=0.01):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(parameters, lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(parameters, lr=lr) 
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(parameters, lr=lr)
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(parameters, lr=lr)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim
    
    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "crossentropy":
                loss_func = F.cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    
