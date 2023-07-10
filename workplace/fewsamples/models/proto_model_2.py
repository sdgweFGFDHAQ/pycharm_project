# encoding=utf-8
import torch
from torch import nn
from icecream.icecream import ic

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProtoTypicalNet2(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_labels, dropout=0.5, requires_grad=False):
        super(ProtoTypicalNet2, self).__init__()
        self.input_dim = embedding_dim
        self.hidden_dim = hidden_dim
        num_labels = 1
        self.num_labels = num_labels

        # 线性层进行编码
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding, requires_grad=requires_grad)

        # 原型网络核心
        self.proto_point = nn.Parameter(torch.randn(num_labels, hidden_dim))

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)

        self.prototype = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.last = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_labels, num_labels),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        embedding_inputs = self.embedding(inputs)
        embedding_inputs = embedding_inputs.to(torch.float32)

        x_inputs, _ = self.lstm(embedding_inputs)
        x = x_inputs[:, -1, :]

        x_pt = self.prototype(x)
        distances = torch.cdist(x_pt, self.proto_point)

        output = self.last(distances)
        return output

    # def forward(self, s_inputs, label, q_inputs):
    #     support_embedding = self.embedding(s_inputs)
    #     s_inputs = support_embedding.to(torch.float32)
    #     s_x, _ = self.lstm(s_inputs)
    #     s_feature = s_x[:, -1, :]
    #
    #     query_embedding = self.embedding(q_inputs)
    #     q_inputs = query_embedding.to(torch.float32)
    #     q_x, _ = self.lstm(q_inputs)
    #     q_feature = q_x[:, -1, :]
    #
    #     # # 提取特征
    #     label_embedding = label.unsqueeze(2) * self.label_embedding.unsqueeze(0)
    #     # e 为标签在该样本下的向量表示,标签是one-hot，不用求和
    #     # e = torch.sum(torch.tan(g(embedding) * g(label)), dim=0)  # 6*5
    #     e = torch.sum(torch.sin(s_feature.unsqueeze(1).repeat(1, self.num_class, 1) * label_embedding), dim=0)
    #     # 将0值所在位置替换为负无穷大
    #     # a 为计算得到的样本权重
    #     a = torch.softmax(e, dim=0)
    #     # 计算原型表示
    #     c = 0.5 * torch.sum(a.unsqueeze(0) * s_feature.unsqueeze(1).repeat(1, self.num_class, 1), dim=0) \
    #         + 0.5 * self.label_embedding
    #
    #     # 计算查询集标签到原型点的距离
    #     distances = torch.sqrt(torch.sum((c.unsqueeze(0) - q_feature.unsqueeze(1)) ** 2, dim=2))
    #     output = self.prototype(-distances)
    #     return output
