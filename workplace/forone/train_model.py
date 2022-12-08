# -*- coding: utf-8 -*-
import math
import operator
from collections import defaultdict

import jieba.analyse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB


def tf_idf_by_python(list_words):
    # 总词频统计
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1

    # 计算每个词的TF值
    word_tf = {}  # 存储每个词的tf值
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

    # 计算每个词的IDF值
    doc_num = len(list_words)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i] += 1
    for i in doc_frequency:
        word_idf[i] = math.log(doc_num / (word_doc[i] + 1))

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i] * word_idf[i]

    # 对字典按值由大到小排序
    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_feature_select


def tf_iwf_by_python(list_words):
    # 总词频统计
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1

    # 计算每个词的TF值
    word_tf = {}  # 存储每个词的tf值
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

    # 计算每个词的IDF值
    doc_num = len(list_words)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i] += 1
    for i in doc_frequency:
        word_idf[i] = math.log(doc_num / (word_doc[i] + 1))

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i] * word_idf[i]

    # 对字典按值由大到小排序
    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_feature_select


# wordlists只有一个文档，那IDF？
def tf_idf_by_jieba(csv_data):
    wordlists = [csv_data['cut_name'].values]
    print(wordlists)
    category = jieba.analyse.extract_tags(wordlists, topK=10, withWeight=False, allowPOS=())
    print(category)


# 朴素贝叶斯训练模型
def b_train_parameter(X, y):
    nb_model = BernoulliNB()
    nb1 = CategoricalNB(alpha=1)
    nb2 = MultinomialNB()
    nb3 = ComplementNB()
    # scores = cross_val_score(nb_model, X, y, cv=10, scoring='accuracy')
    # print('Accuracy:{:.4f}'.format(scores.mean()))
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    transfer = TfidfTransformer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)
    for model in [nb_model, nb2, nb3]:
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        print("准确率为:", score)
