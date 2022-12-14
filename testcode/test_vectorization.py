import logging
import time
from ast import literal_eval
import sys
import os
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import numpy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import ComplementNB, BernoulliNB, MultinomialNB
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from multiprocessing import Manager, Pool
from workplace.forone.count_category_num import feature_vectorization
from workplace.forone.mini_tool import cut_word
from testcode.test_pool import pool_test

def test_transform():
    csv_data = pd.read_csv('./aaa.csv', usecols=['name', 'category3_new', 'cut_name'], nrows=5)
    csv_data['cut_name'] = csv_data['cut_name'].apply(literal_eval)
    lll = []
    for data in csv_data['cut_name']:
        list1 = [str(i) for i in data]
        str_data = ' '.join(list1)
        lll.append(str_data)
    print(lll)
    vectorized = CountVectorizer()
    toarray = vectorized.fit_transform(lll).astype(np.int8).toarray()
    print(toarray)
    print(csc_matrix(toarray).sum(axis=0))
    s3 = time.localtime(time.time())


def test_getCategory():
    csv_data = pd.read_csv('aaa.csv', usecols=['name', 'category3_new', 'cut_name'])
    csv_data['cut_name'] = csv_data['cut_name'].apply(literal_eval)
    dummy = feature_vectorization(csv_data)
    print(dummy)
    classif = mutual_info_classif(dummy, csv_data['category3_new'], discrete_features=True)
    igr_list = dict(zip(dummy.columns, classif))
    print(igr_list)


def test_dict_遍历性能():
    dict1 = {}
    dict0 = dict()
    for j in range(88):
        dict1['类别' + str(j)] = j - 1
    for i in range(10000):
        dict0['特征' + str(i)] = dict1
    print(dict0)
    list1 = []
    list0 = {}
    for n in range(10000):
        list1.append('特征' + str(n))
    for m in range(88):
        list0['类别' + str(m)] = list1
    print(list0)
    print(time.localtime(time.time()))
    for a, b in dict0.items():
        for c, d in b.items():
            list_x = list0[c]
            if a in list_x:
                continue
    print(time.localtime(time.time()))


def test_dict():
    dict_list = [{'a1': 1},{'a2': 2},{'a3': 3},{'a4': 4}]
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
    copy = manager_dict['a']|new_dict
    manager_dict['a'] = copy
    print("{}:{}".format(os.getpid(), manager_dict))


if __name__ == '__main__':
    # test_transform()
    # test_dict_遍历性能()
    double_dict = test_dict()


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
