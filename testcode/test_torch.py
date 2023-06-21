# -*- encoding:utf-8 -*-
import math
import random

import jieba
import numpy as np
import pandas as pd
from icecream import ic
import torch
from torch import nn
from transformers import BertTokenizer

import transformers

if __name__ == '__main__':
    print(torch.__version__)  # 1.13.0
    print(transformers.__version__)  # 3.1.0


def ic_test():
    a = 0
    ic(a)


def dim():
    batch_size = 3
    hidden_size = 5
    embedding_dim = 6
    seq_length = 4
    num_layers = 1
    num_directions = 1
    vocab_size = 20

    input_data = np.random.uniform(0, 19, size=(batch_size, seq_length))
    input_data = torch.from_numpy(input_data).long()
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    lstm_layer = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                               bias=True, batch_first=False, dropout=0.5, bidirectional=False)
    lstm_input = embedding_layer(input_data)
    assert lstm_input.shape == (batch_size, seq_length, embedding_dim)
    a, (b, c) = lstm_layer(lstm_input)
    print(a)
    lstm_input.transpose_(1, 0)
    assert lstm_input.shape == (seq_length, batch_size, embedding_dim)
    output, (h_n, c_n) = lstm_layer(lstm_input)
    print(output)
    assert output.shape == (seq_length, batch_size, hidden_size)
    assert h_n.shape == c_n.shape == (num_layers * num_directions, batch_size, hidden_size)


class ffs(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forword(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

    def __call__(self, *args, **kwargs):
        ic('call')


class A:
    def __init__(self, a):
        self.a = a
        print('A.a')

    def aaa(self):
        print('aaa')


class B(A):
    def __init__(self, b):
        super(A, self).__init__()
        self.b = b
        print('B.b')


def classandmodel():
    input_size = 20
    hidden_sizes = [10, 5]
    output_size = 2
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.Softmax(dim=1))
    ac = B(2)
    print(ac.b)
    print(ac.aaa())


def nD2nD():
    # 定义模型输出为 n_classes 个类别的概率分布
    output = torch.randn(3, 5)
    print(output)
    # 定义实际标签
    target = torch.empty(3, dtype=torch.long).random_(5)

    # 将标签张量降维为1D
    target = torch.squeeze(target)
    print(target)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 计算损失
    loss = criterion(output, target)
    print(loss)


def crossE():
    criterion = nn.CrossEntropyLoss()
    output = torch.randn(3, 5, requires_grad=True)
    label = torch.empty(3, dtype=torch.long).random_(5)
    loss = criterion(output, label)

    print("网络输出为3个5类:")
    print(output)
    print("要计算loss的类别:")
    print(label)
    print("计算loss的结果:")
    print(loss)

    first = [0, 0, 0]
    for i in range(3):
        first[i] = -output[i][label[i]]
    second = [0, 0, 0]
    for i in range(3):
        for j in range(5):
            second[i] += math.exp(output[i][j])
    res = 0
    for i in range(3):
        res += (first[i] + math.log(second[i]))
    print("自己的计算结果：")
    print(res / 3)


def test_a(df_row, index):
    spilt = df_row['cut_name'].split()
    index.append(spilt)
    df_r = df_row.copy()
    for w in spilt:
        df_r['cut_name'] = w
        df_row = pd.concat([df_row, df_r], axis=1)
    return df_row


if __name__ == '__main__':
    test_a()
