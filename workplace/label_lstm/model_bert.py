# encoding=utf-8
from icecream import ic
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BertLSTMNet(nn.Module):
    def __init__(self, bert_embedding, input_dim, hidden_dim, num_classes, num_layers, dropout=0.5, requires_grad=False):
        super(BertLSTMNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        # embedding layer 线性层进行编码
        self.bert_embedding = bert_embedding
        for param in self.bert_embedding.parameters():
            param.requires_grad = requires_grad
        # 解冻后面1层的参数
        for param in self.bert_embedding.encoder.layer[-1:].parameters():
            param.requires_grad = True

        # 将一个不可训练的类型为Tensor的参数转化为可训练的类型为parameter的参数，并将这个参数绑定到module里面，成为module中可训练的参数。
        # self.embedding.weight = torch.nn.Parameter(embedding, requires_grad=requires_grad)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim * 2, num_classes),
                                        nn.Sigmoid())

    def forward(self, inputs):
        inputs_bert = self.bert_embedding(inputs).last_hidden_state[:, 0]
        inputs_bert = inputs_bert.to(torch.float32)
        x, _ = self.lstm(inputs_bert)
        # 取用 LSTM 最后一个的 hidden state
        x = x[:, -1, :]
        x = self.classifier(x)
        return x
