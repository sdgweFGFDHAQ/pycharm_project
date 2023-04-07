# encoding=utf-8
from icecream import ic
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTMNet(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_classes, num_layers, dropout=0.5, requires_grad=True):
        super(LSTMNet, self).__init__()
        # embedding layer
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        # 将一个不可训练的类型为Tensor的参数转化为可训练的类型为parameter的参数，并将这个参数绑定到module里面，成为module中可训练的参数。
        self.embedding.weight = torch.nn.Parameter(embedding, requires_grad=requires_grad)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim * 2, num_classes),
                                        nn.Sigmoid())

    def forward(self, inputs):
        # nonzero_idxs = torch.nonzero(inputs)
        # input_lengths = torch.bincount(nonzero_idxs[:, 0])
        # print(input_lengths)

        # (batch_size * len_sen(句子长度) * input_size(单词的向量维度)
        # inputs = self.dropout(self.embedding(inputs))

        # rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        inputs = self.embedding(inputs)

        # x:batch_size * len_sen * hidden_size
        inputs = inputs.to(torch.float32)
        # ic(inputs)
        # inputs = rnn.pack_padded_sequence(inputs, lengths=input_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(inputs, None)
        # x, _ = rnn.pad_packed_sequence(x, batch_first=True)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最后一个的 hidden state
        x = x[:, -1, :]
        x = self.classifier(x)
        return x

    # def forward(self, input1, input2):
    #     out1 = self.forward_once(input1)
    #     out2 = self.forward_once(input2)
    #     return out1, out2
