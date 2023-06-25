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
        # 解冻后面3层的参数
        for param in self.bert_embedding.encoder.layer[-1:].parameters():
            param.requires_grad = True

        # 原型网络核心
        self.prototype = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),

            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_class),
        )

        # 用于改变维度大小
        # self.linear = nn.Linear(hidden_dim, self.num_class)
        self.last = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_class, num_class),
            nn.Sigmoid())

    def forward(self, support_input, support_label, query_input):
        # # 由于版本原因，当前选择的bert模型会返回tuple，包含(last_hidden_state,pooler_output)
        support_embedding = self.bert_embedding(support_input).last_hidden_state[:, 0]
        query_embedding = self.bert_embedding(query_input).last_hidden_state[:, 0]
        support_point = self.prototype(support_embedding)
        query_point = self.prototype(query_embedding)

        # 提取特征
        # e 为标签在该样本下的向量表示,标签是one-hot，不用求和
        # e = torch.sum(torch.tan(g(embedding) * g(label)), dim=0)  # 6*5
        e = torch.tan(support_point * support_label)
        # 将0值所在位置替换为负无穷大
        # e[e == 0] = float('-inf')
        f = torch.where(e == 0, float('-inf'), e)

        # a 为计算得到的样本权重
        a = torch.softmax(f, dim=0)
        # 计算原型表示
        # c = b * torch.matmul(a.t(), embedding) + (1 - b) * label.t()
        c = torch.matmul(a.t(), support_point)
        # 计算查询集标签到原型点的距离
        distances = torch.sqrt(torch.sum((c.unsqueeze(0) - query_point.unsqueeze(1)) ** 2, dim=2))

        d_output = self.last(distances)
        return d_output
