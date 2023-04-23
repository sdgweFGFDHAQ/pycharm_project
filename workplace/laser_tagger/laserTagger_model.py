# encoding=utf-8
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class laserTaggerModel(nn.Module):
    def __init__(self, hidden_dim, num_tags, dropout=0.1):
        super(laserTaggerModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # num_tags 词汇数量
        self.linear = nn.Linear(hidden_dim, num_tags)
        self.linear.weight = nn.Parameter(nn.init.normal())

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        outputs = self.linear(inputs)
        return outputs
