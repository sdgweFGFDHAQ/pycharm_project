import logging
import os
import time

from ast import literal_eval

import torch
from gensim.models import Word2Vec, KeyedVectors
import jieba
from icecream import ic
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from multiprocessing import Manager, Pool
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from workplace.label_nb.count_category_num import feature_vectorization
from keras.utils import pad_sequences


def test_dict():
    dict_list = [{'a1': 1}, {'a2': 2}, {'a3': 3}, {'a4': 4}]
    series = pd.Series(['a', 'b', 'c'])
    print(series.values)
    manager_dict = Manager().dict()
    for i in series.values:
        manager_dict[i] = dict()
    print(manager_dict)
    pool = Pool(processes=4)
    for i in range(4):
        pool.apply_async(use_update, args=(dict_list[i], manager_dict))
    pool.close()
    pool.join()
    print("就此结束", manager_dict)
    return manager_dict


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
    a = torch.randn(3, 5)
    print(a)
    b = torch.randn(3, 5)
    print(b)
    cosine_value = torch.cosine_similarity(a, b, dim=1)
    print(cosine_value)

    support_size0 = a.shape[0]
    class_meta_dict = {}
    class_meta_dict[0] = torch.sum(a, dim=0) / 3

    class_meta_information = torch.zeros(size=[len(class_meta_dict), a.shape[1]])
    for key, item in class_meta_dict.items():
        class_meta_information[key, :] = class_meta_dict[key]
    ic(class_meta_information)
    ic(b)
    N_query = b.shape[0]
    result = torch.zeros(size=[N_query, 1])
    for i in range(0, N_query):
        temp_value = b[i].repeat(1, 1)
        cosine_value = torch.cosine_similarity(class_meta_information, temp_value, dim=1)
        result[i] = cosine_value
    ic(result)
