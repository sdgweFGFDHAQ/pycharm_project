# encoding=utf-8
import torch
from torch import nn
import torch.nn.functional as F
from icecream.icecream import ic

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProtoTypicalNet2(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_class, dropout=0.5, beta=0.5, requires_grad=False):
        super(ProtoTypicalNet2, self).__init__()
        self.input_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.beta = beta

        # 线性层进行编码
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding, requires_grad=requires_grad)
        # 对多标签编码
        self.label_embedding = torch.normal(0, 1, size=(num_class, hidden_dim * 2))
        # 原型网络核心
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)

        self.prototype = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_class),
            nn.Sigmoid()
        )

    def forward(self, inputs, label):
        support_embedding = self.embedding(inputs)
        s_inputs = support_embedding.to(torch.float32)
        s_x, _ = self.lstm(s_inputs)
        output_point = s_x[:, -1, :]

        if label is None:
            output = self.prototype(output_point)
        else:
            # # 提取特征
            label_embedding = label.unsqueeze(2) * self.label_embedding.unsqueeze(0)
            # e 为标签在该样本下的向量表示,标签是one-hot，不用求和
            # e = torch.sum(torch.tan(g(embedding) * g(label)), dim=0)  # 6*5
            e = torch.sum(torch.sin(output_point.unsqueeze(1).repeat(1, self.num_class, 1) * label_embedding), dim=0)
            # 将0值所在位置替换为负无穷大
            # f = torch.where(e == 0, float('-inf'), e)
            # a 为计算得到的样本权重
            a = torch.softmax(e, dim=0)
            # 计算原型表示
            # c = b * torch.matmul(a.t(), embedding) + (1 - b) * label.t()
            c = 0.5 * torch.sum(a.unsqueeze(0) * output_point.unsqueeze(1).repeat(1, self.num_class, 1), dim=0) \
                + 0.5 * self.label_embedding
            output = self.prototype(c.unsqueeze(0))
        # ic(result)
        return output
