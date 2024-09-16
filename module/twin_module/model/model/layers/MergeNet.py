import torch

import torch.nn as nn

class MergeNet(nn.Module):
    def __init__(self, left_emb_size, right_emb_size, hidden_size, n_layer, class_num, use_bias=True):
        super(MergeNet, self).__init__()
        self.left_layer  = nn.Linear(left_emb_size, hidden_size)
        self.right_layer = nn.Linear(right_emb_size, hidden_size)
        block_list = []
        input_size = 2*hidden_size
        for index in range(n_layer):
            if index+1 == n_layer:
                # 最后二分类层输出
                output_size = class_num
            else:
                output_size = input_size//2
            block_list.extend([nn.LayerNorm([input_size]), nn.Linear(input_size, output_size, bias=use_bias)])
            input_size = output_size
        self.merge_block = nn.ModuleList(block_list)
        #self.out = nn.Linear(input_size, output_size, bias=use_bias)

    def forward(self, left_embedding, right_embedding):
        left_embedding  = self.left_layer(left_embedding)
        right_embedding = self.right_layer(right_embedding)
        for index, layer in enumerate(self.merge_block):
            if index == 0:
                x = torch.relu(layer(torch.cat((left_embedding, right_embedding), dim=-1)))
            elif index%2 == 1:
                # 线性层
                x = layer(x)
            else:
                # 归一化层
                x = torch.relu(layer(x))
        #output = self.out(x)
        return x
