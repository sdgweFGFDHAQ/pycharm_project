# encoding=utf-8
import torch
from torch import nn
import torch.nn.functional as F
from icecream.icecream import ic

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProtoTypicalNet(nn.Module):
    def __init__(self, bert_layer, input_dim, hidden_dim, num_class, dropout=0.5, beta=0.5, requires_grad=False):
        super(ProtoTypicalNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.beta = beta

        # 线性层进行编码
        self.bert_embedding = bert_layer
        for param in self.bert_embedding.parameters():
            param.requires_grad = requires_grad
        # 解冻后面1层的参数
        for param in self.bert_embedding.encoder.layer[-1:].parameters():
            param.requires_grad = True

        # 原型网络核心
        self.proto_point = nn.Parameter(torch.randn(num_class, hidden_dim))

        self.prototype = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim)
        )

        self.last = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_class, num_class),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs_embedding = self.bert_embedding(inputs).last_hidden_state[:, 0]

        x_pt = self.prototype(inputs_embedding)
        distances = torch.cdist(x_pt, self.proto_point)
        output = self.last(distances)

        return output
