import logging
import os
import time

from ast import literal_eval
from gensim.models import Word2Vec, KeyedVectors
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


def fit_model_by_deeplearn():
    df = pd.read_csv('aaa.csv')
    tokenizer = Tokenizer(num_words=20, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    sample_lists = list()
    for i in df['cut_name']:
        i = literal_eval(i)
        sample_lists.append(' '.join(i))
    tokenizer.fit_on_texts(sample_lists)
    word_index = tokenizer.word_index
    print('共有 %s 个不相同的词语.' % len(word_index))
    X = tokenizer.texts_to_sequences(sample_lists)
    # 填充X,让X的各个列的长度统一
    X = pad_sequences(X, maxlen=6)
    # # 多类标签的onehot展开
    Y = pd.get_dummies(df['cat_id']).values
    # 拆分训练集和测试集
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    # print(X_train.shape, Y_train.shape)
    # print(X_test.shape, Y_test.shape)
    # 定义模型
    model = Sequential()
    model.add(Embedding(22, 10, input_length=X.shape[1], name='emb'))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(units=64, dropout=0.3, recurrent_dropout=0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X, Y, epochs=5, batch_size=32, validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    print(model.layers[0].output)
    # accuracy = model.evaluate(X_test, Y_test)
    # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accuracy[0], accuracy[1]))


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


if __name__ == '__main__':
    # fit_model_by_deeplearn()
    df = pd.read_csv('../workplace/fewsamples/few_shot.csv', index_col=0)
    token = Tokenizer()
    word_lists = df['cut_name'].values
    token.fit_on_texts(word_lists)
    print(token.word_index)
    seq = token.texts_to_sequences(word_lists)
    x = pad_sequences(seq, maxlen=2)
    y = pd.get_dummies(df['category3_new']).values
    bls = BorderlineSMOTE(k_neighbors=1, kind='borderline-1')
    x_n, y_n = bls.fit_resample(x, y)
    print(x_n)
    xx = [a[0] for a in x_n]
    yy = [b[1] for b in x_n]
    # zz = [c[2] for c in x]
    ax = plt.subplot(1, 1, 1)
    ax.scatter(xx, yy, c=[np.argmax(y_n, axis=1)], cmap='viridis', alpha=0.2)
    plt.show()
