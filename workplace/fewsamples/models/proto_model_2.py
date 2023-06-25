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
        self.label_embedding = nn.Embedding(num_class, embedding_dim)
        # 原型网络核心
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)

        self.prototype = nn.Sequential(nn.Dropout(dropout),
                                       nn.Linear(hidden_dim * 2, num_class))

        self.last = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_class, num_class),
            nn.Sigmoid())

        # 用于改变维度大小
        self.linear = nn.Linear(num_class, num_class)

    def forward(self, support_input, support_label, query_input):
        # # 由于版本原因，当前选择的bert模型会返回tuple，包含(last_hidden_state,pooler_output)
        support_embedding = self.embedding(support_input)
        query_embedding = self.embedding(query_input)

        s_inputs = support_embedding.to(torch.float32)
        s_x, _ = self.lstm(s_inputs, None)
        s_x = s_x[:, -1, :]
        q_inputs = query_embedding.to(torch.float32)
        q_x, _ = self.lstm(q_inputs, None)
        q_x = q_x[:, -1, :]

        support_point = self.prototype(s_x)
        query_point = self.prototype(q_x)

        # # 提取特征
        # e 为标签在该样本下的向量表示,标签是one-hot，不用求和
        # e = torch.sum(torch.tan(g(embedding) * g(label)), dim=0)  # 6*5
        e = torch.tan(support_point * support_label)
        # 将0值所在位置替换为负无穷大
        # f = torch.where(e == 0, float('-inf'), e)
        # a 为计算得到的样本权重
        a = torch.softmax(e, dim=0)
        # 计算原型表示
        # c = b * torch.matmul(a.t(), embedding) + (1 - b) * label.t()
        c = torch.matmul(a.t(), support_point)
        # 计算查询集标签到原型点的距离
        distances = torch.sqrt(torch.sum((c.unsqueeze(0) - query_point.unsqueeze(1)) ** 2, dim=2))
        # sqs = torch.concat((support_point, query_point, support_point - query_point), dim=1)

        # distances = torch.arctan(distances)
        result = self.last(distances)
        return result
