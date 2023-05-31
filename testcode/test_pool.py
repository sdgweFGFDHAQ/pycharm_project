import os
from multiprocessing import Pool, Manager
import re
import jieba
import ast

import pandas as pd
import torch
from sklearn import preprocessing


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


if __name__ == '__main__':
    # csv = pd.read_csv('aaa.csv', index_col=0)
    # alist = Manager().list()
    # pool = Pool(4)
    # ddf = pd.DataFrame()
    # for i in range(5):
    #     pool.apply_async(func=get_df, args=(csv, i, alist))
    # pool.close()
    # pool.join()
    # concat = pd.concat(alist, ignore_index=True)
    # print(concat)
    a = [1, 2, 3, 4]
    b = a.copy()
    query_list = [a.pop(), b.pop()]
    print(query_list)
