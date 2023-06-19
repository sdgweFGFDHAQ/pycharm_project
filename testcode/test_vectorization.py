import logging
import os
import time

from ast import literal_eval

from multiprocessing import Manager, Pool
import numpy as np
import pandas as pd
import torch
from icecream import ic
from torch import nn


def use_update(new_dict, manager_dict):
    # manager_dict['a'].update(new_dict)
    copy = manager_dict['a'] | new_dict
    manager_dict['a'] = copy
    print("{}:{}".format(os.getpid(), manager_dict))


def log_record():
    # 创建日志对象(不设置时默认名称为root)
    log = logging.getLogger()
    # 设置日志级别(默认为WARNING)
    log.setLevel('INFO')
    # 设置输出渠道(以文件方式输出需设置文件路径)
    file_handler = logging.FileHandler('test.log', encoding='utf-8')
    file_handler.setLevel('INFO')
    # 设置输出格式(实例化渠道)
    fmt_str = '%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt_str)
    # 绑定渠道的输出格式
    file_handler.setFormatter(formatter)
    # 绑定渠道到日志收集器
    log.addHandler(file_handler)
    return log


class sdas():
    def __init__(self):
        pass

    def __iter__(self):
        na = ['aaa.csv']
        for i in na:
            df = pd.read_csv(i)
            df['cut_name'] = df['cut_name'].apply(literal_eval)
            for j in df['cut_name'].values:
                yield j


def dadsfs():
    name_iter = ['小年轻 伙子', '老正经 大爷', '偷偷 象棋', '老正经 伙子', '小年轻 大爷', '偷偷 围棋', '小年轻 象棋',
                 '老正经 围棋', '小赌 一手']
    name_list = list()
    for name in name_iter:
        name_list.append(name.split())
    vec = Word2Vec(sentences=name_list, vector_size=5, min_count=2, window=2, workers=1, sg=1, epochs=5)
    vec.save('2vec.model')
    name_list = [['车', '打', '炮'], ['炮', '吃', '马'], ['马', '跳', '车']]
    vec.build_vocab(name_list, update=True)
    vec.train(name_list, total_examples=vec.corpus_count, epochs=5)
    vec.wv.save_word2vec_format('2vec.vector')


def wdf():
    cs = pd.read_csv('../workplace/fewsamples/data/few_shot.csv')
    psd = cs['cut_name'].values
    sda_list = []
    for ab in psd:
        sda_list.append(len(ab.split()))
    print(sda_list)


def read_train():
    with open('train.txt', encoding='utf-8', mode='w') as tf:
        with open('data.train', encoding='utf-8', mode='r') as f:
            for readline in f:
                split = readline.rstrip('\n').replace('\ufeff', '').split()
                if split[1] != '0' and len(split) >= 4:
                    tf.write(split[2] + ' ' + split[3] + '\n')


if __name__ == '__main__':
    # read_train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 6 * 12
    label = torch.tensor([[1, 1, 0, 1, 0],
                          [0, 1, 0, 1, 1],
                          [1, 0, 1, 1, 0],
                          [1, 1, 0, 1, 0],
                          [0, 1, 0, 1, 1],
                          [1, 0, 1, 0, 0]]).to(device, dtype=torch.float)
    embedding = torch.tensor([[0.89, 0.66, 0.33, 0.56, 0.23, 0.37, 0.81, 0.66, 0.23, 0.56, 0.33, 0.39],
                              [0.02, 0.43, 0.91, 0.43, 0.65, 0.21, 0.89, 0.66, 0.33, 0.56, 0.23, 0.37],
                              [0.77, 0.54, 0.65, 0.16, 0.42, 0.37, 0.87, 0.66, 0.23, 0.56, 0.23, 0.37],
                              [0.89, 0.66, 0.23, 0.56, 0.23, 0.37, 0.89, 0.66, 0.22, 0.56, 0.23, 0.35],
                              [0.12, 0.43, 0.99, 0.42, 0.66, 0.31, 0.89, 0.66, 0.23, 0.52, 0.23, 0.37],
                              [0.71, 0.54, 0.65, 0.16, 0.43, 0.35, 0.89, 0.66, 0.23, 0.51, 0.21, 0.37]])
    y_label = torch.tensor([[1, 1, 0, 1, 0],
                          [0, 1, 0, 1, 1],
                          [1, 0, 1, 1, 0],
                          [1, 1, 0, 1, 0],
                          [0, 1, 0, 1, 1],
                          [1, 0, 1, 0, 0]])
    embedding_size, hidden_size = 12, 5
    input_size, output_size = embedding_size, hidden_size
    # 降维，768用隐层降小一点
    hidden_layer = nn.Linear(input_size, output_size)

    # 提取特征
    g = nn.Linear(input_size, output_size)
    # 计算e 为标签在该样本下的向量表示,标签是one-hot，不用求和
    # e = torch.sum(torch.tan(g(embedding) * g(label)), dim=0)  # 6*5
    dc = g(embedding) * label  # 6*5
    e = torch.tan(dc)
    ic(e)
    # 计算样本权重,将0值所在位置替换为负无穷大
    e[e == 0] = float('-1e-9')
    ic(e)
    a = torch.softmax(e, dim=0)
    ic(a)
    # 计算原型表示
    b = 0.5
    # c = b * torch.matmul(a.t(), embedding) + (1 - b) * label.t()
    c = b * torch.matmul(a.t(), hidden_layer(embedding))
    print(c)
    # 计算查询集标签到原型点的距离
    distances = torch.sqrt(torch.sum((c.unsqueeze(0) - y_label.unsqueeze(1)) ** 2, dim=2))
    print(distances)
