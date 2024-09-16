import torch.nn as nn
import torch.nn.functional as F


class CatLayer(nn.Module):
    def __init__(self, input_size, output_size, n_layer, use_bias=True, **kwargs):
        super(CatLayer, self).__init__()
        self.norm_layer = nn.Linear(input_size, output_size, bias=use_bias)
        self.func  = nn.LeakyReLU(0.5)
        self.layer_blocks = nn.ModuleList([nn.Linear(output_size, output_size, bias=use_bias) for _ in range(n_layer)])
    
    def forward(self, x):
        #print("输入进来的数据 device；", x.device)
        #print("当前层的 device: ", self.norm_layer.device)
        embedding = self.func((F.normalize(self.norm_layer(x), p=2, dim=-1)))
        #print("可以到这里")
        for layer in self.layer_blocks:
            embedding = self.func((F.normalize(layer(embedding), p=2, dim=-1)))
        #print("线形层成功结束")
        return embedding
