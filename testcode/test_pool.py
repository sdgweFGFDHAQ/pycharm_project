import math
import os
from multiprocessing import Pool, Manager
import re
import jieba
import ast

import numpy
import pandas as pd
import torch
from sklearn import preprocessing
from icecream.icecream import ic
from torch import nn


def resub():
    word = "发as|sdf35%^丨&*但是的风格aad$"
    word = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]|[丨]', '', word)
    print(word)
    word = "[dsgfd]g[dff]]"
    word = re.sub(r'\[|\]', '', word)
    print(word)
    word = "EXO是x的是X的excelAxB"
    word = re.sub(r'(?<=[\u4e00-\u9fa5])(x|X)(?=[\u4e00-\u9fa5])|(?<=[A-B])x(?=[A-B])', '', word)
    print(word)


def evaltest():
    value = {'烤面包': 8.755, '系列': 9.142, '兰姐': 9.152, '旺城': 9.172, '莱': 9.442, '茂映': 9.342, '百家': 8.736}
    scale = preprocessing.minmax_scale(list(value.values()))
    value = dict(zip(value.keys(), scale))
    print(value)


def pool_test():
    pool = Pool(processes=4)
    c = Manager().list()
    a = 1
    b = 2
    for i in range(4):
        pool.apply_async(co_num, args=(a, b, c))
    pool.close()
    pool.join()
    print(c)
    return c


def co_num(a, b, c):
    a_b = a + b
    c.append(a_b)
    return c


def get_df(df, i, al):
    print(i)
    al.append(df.head(1))


def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    ic(y_pred)
    y_pred_neg = y_pred - y_true * 1e12
    ic(y_pred_neg)
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    ic(y_pred_pos)
    zeros = torch.zeros_like(y_pred[..., :1])

    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


if __name__ == '__main__':
    # crossentropy = multilabel_categorical_crossentropy(
    #     torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 0]]),
    #     torch.tensor([[0.8, 0.6, 0.9], [0.7, -0.2, 0.1], [0.2, -0.6, -0.7]])
    # )
    # print(crossentropy)

    criterion = nn.BCELoss(reduction='mean')
    ou = criterion(
        torch.tensor([[1., 1., 1.], [1., 0., 1.], [0., 0., 0.]]),
        torch.tensor([[1., 1., 1.], [1., 1., 1.], [0., 0., 0.]])
    )
    print(ou)
