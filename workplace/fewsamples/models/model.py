# encoding=utf-8

import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTMNet(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_classes, num_layers, dropout=0.5, requires_grad=True):
        super(LSTMNet, self).__init__()
        # embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        # 将一个不可训练的类型为Tensor的参数转化为可训练的类型为parameter的参数，并将这个参数绑定到module里面，成为module中可训练的参数。
        self.embedding.weight = torch.nn.Parameter(embedding, requires_grad=requires_grad)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim * 2, num_classes),
                                        nn.Sigmoid())

    def forward(self, inputs):
        # (batch_size * len_sen(句子长度) * input_size(单词的向量维度)
        # inputs = self.dropout(self.embedding(inputs))
        inputs = self.embedding(inputs)
        # x:batch_size * len_sen * hidden_size
        inputs = inputs.to(torch.float32)
        # print(inputs)
        x, _ = self.lstm(inputs, None)

        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最后一个的 hidden state
        x = x[:, -1, :]
        # print("x", x)
        x = self.classifier(x)
        return x
