# -*- coding: utf-8 -*-
import jieba.analyse
from collections import defaultdict
import math
import operator
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from workplace.forone.count_category_num import count_the_number_of_categories


def loadDataSet():
    dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表好，0代表不好
    return dataset, classVec


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


# wordlists只有一个文档，那IDF？
def tf_idf_by_jieba(csv_data):
    wordlists = [csv_data["word_name"].values]
    print(wordlists)
    category = jieba.analyse.extract_tags(wordlists, topK=10, withWeight=False, allowPOS=())
    print(category)


# 贝叶斯测试
def nb_test():
    X, y = load_breast_cancer().data, load_breast_cancer().target
    # one-hot
    enc = preprocessing.OneHotEncoder()
    enc.fit(X)
    X = enc.transform(X).toarray()
    # pca降维
    # pca = PCA(n_components=3)  # 从5列降到3列
    # pca.fit(X)
    # X = pca.transform(X)
    print(X)
    nb1 = GaussianNB()
    nb2 = MultinomialNB()
    nb3 = BernoulliNB()
    nb4 = ComplementNB()
    for model in [nb1, nb2, nb3, nb4]:
        scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
        print("Accuracy:{:.4f}".format(scores.mean()))


# 朴素贝叶斯训练模型
def b_train_parameter(csv_data):
    X, y = count_the_number_of_categories(csv_data), csv_data["type"]
    nb_model = BernoulliNB()
    scores = cross_val_score(nb_model, X, y, cv=10, scoring="accuracy")
    print("Accuracy:{:.4f}".format(scores.mean()))


if __name__ == '__main__':
    # data_list, label_list = loadDataSet()  # 加载数据
    # features = tf_idf_by_python(data_list)  # 所有词的TF-IDF值
    # print(features)
    # print(len(features))
    nb_test()
