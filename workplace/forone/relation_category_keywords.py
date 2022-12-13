import pandas as pd
import re
import time
import math
import operator
from collections import defaultdict
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
import ast


def get_feature_prob(X, y):
    c_nb = ComplementNB()
    c_nb.fit(X, y)
    feature_prob = pd.DataFrame(c_nb.feature_log_prob_, index=c_nb.classes_, columns=X.columns)
    to_dict = feature_prob.to_dict(orient='index')
    keyword_dict = pd.read_csv('../keyword_dict.csv')
    for key, value in to_dict.items():
        if key not in keyword_dict['category'].values:
            pd.concat([keyword_dict, pd.DataFrame({'category': key, 'keyword': list(value.keys())})], axis=0)
            # keyword_dict.append({'category': key, 'keyword': list(value.keys())}, ignore_index=True)
        mean = np.mean(list(value.values()))
        ndarray_values = str(keyword_dict.loc[keyword_dict['category'] == key, 'keyword'].values)
        values = re.sub(r"\[|\]|'|\"", '', ndarray_values).split(',')
        for i_key in values:
            if i_key not in value.keys():
                value[i_key] = mean + 0.1
        value = dict(sorted(value.items(), key=lambda x: (float(x[1])), reverse=True))
        new_value = {}
        keys = value.keys()
        if len(keys) > 100:
            for k in list(keys)[0:int(0.4 * len(keys))]:
                new_value[k] = value[k]
            to_dict[key] = new_value
    keys = to_dict.keys()
    values = to_dict.values()
    df = pd.DataFrame({'category': keys, 'keyword': values})
    df.to_csv('E:\\testwhat\pyProjects\\testPY\\workplace\\filename.csv', index=False)


def update_keyword(X, y):
    c_nb = ComplementNB()
    c_nb.fit(X, y)
    feature_prob = pd.DataFrame(c_nb.feature_log_prob_, index=c_nb.classes_, columns=X.columns)
    to_dict = feature_prob.to_dict(orient='index')
    keyword_dict = pd.read_csv('E:\\testwhat\pyProjects\\testPY\\workplace\\filename.csv')
    for key, value in to_dict.items():
        if key not in keyword_dict['category'].values:
            keyword_dict.append({'category': key, 'keyword': list(value.keys())}, ignore_index=True)
        mean = np.mean(list(value.values()))
        ndarray_values = str(keyword_dict.loc[keyword_dict['category'] == key, 'keyword'].values)
        # 现在是字典
        values = ast.literal_eval(re.sub(r"\[|\]", '', ndarray_values))
    #     for i_key in values:
    #         if i_key not in value.keys():
    #             value[i_key] = mean
    #     value = dict(sorted(value.items(), key=lambda x: (float(x[1])), reverse=True))
    #     new_value = {}
    #     keys = value.keys()
    #     if len(keys) > 100:
    #         for k in list(keys)[int(0.05 * len(keys)):int(0.4 * len(keys))]:
    #             new_value[k] = value[k]
    #         to_dict[key] = new_value
    # keys = to_dict.keys()
    # values = to_dict.values()
    # df = pd.DataFrame({'category': keys, 'keyword': values})
    # df.to_csv('E:\\testwhat\pyProjects\\testPY\\workplace\\filename.csv', index=False)

def out_key_word():
    print("得到的分类模型为")


def coculate_category():
    print("新数据的类别为")

def forecast_results(X, y):
    c_nb = ComplementNB()
    transfer = TfidfTransformer()
    X = transfer.fit_transform(X)
    c_nb.fit(X, y)
    print(c_nb.predict(X))
    print("准确率为:", c_nb.score(X, y))
    print(c_nb.predict_proba(X))
