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

        # 对多标签编码
        self.label_embedding = torch.normal(0, 1, size=(num_class, input_dim))

        # 原型网络核心
        self.prototype = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_class),
            nn.Sigmoid()
        )

    def forward(self, support_input, support_label, query_input):
        # # 由于版本原因，当前选择的bert模型会返回tuple，包含(last_hidden_state,pooler_output)
        support_embedding = self.bert_embedding(support_input).last_hidden_state[:, 0]
        query_embedding = self.bert_embedding(query_input).last_hidden_state[:, 0]
        label_embedding = support_label.unsqueeze(2) * self.label_embedding.unsqueeze(0)

        # # 提取特征
        # e 为标签在该样本下的向量表示,标签是one-hot，不用求和
        # e = torch.sum(torch.tan(g(embedding) * g(label)), dim=0)  # 6*5
        e = torch.sum(torch.sin(support_embedding.unsqueeze(1).repeat(1, self.num_class, 1) * label_embedding), dim=1)
        # 将0值所在位置替换为负无穷大
        # f = torch.where(e == 0, float('-inf'), e)
        # a 为计算得到的样本权重
        a = torch.softmax(e, dim=0)
        # 计算原型表示
        # c = b * torch.matmul(a.t(), embedding) + (1 - b) * label.t()
        c = torch.matmul(a.t(), support_embedding)
        # 计算查询集标签到原型点的距离
        distances = torch.sqrt(torch.sum((c.unsqueeze(0) - query_embedding.unsqueeze(1)) ** 2, dim=2))
        # sqs = torch.concat((support_point, query_point, support_point - query_point), dim=1)

        # distances = torch.arctan(distances)
        result = self.prototype(distances)
        return result
