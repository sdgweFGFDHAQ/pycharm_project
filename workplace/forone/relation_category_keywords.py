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


def get_feature_prob(X, y):
    c_nb = ComplementNB()
    c_nb.fit(X, y)
    feature_prob = pd.DataFrame(c_nb.feature_log_prob_, index=c_nb.classes_, columns=X.columns)
    to_dict = feature_prob.to_dict(orient='index')
    keyword_dict = pd.read_csv('../keyword_dict.csv')
    for key, value in to_dict.items():
        if key not in keyword_dict['category'].values:
            keyword_dict.append({'category': key, 'keyword': list(value.keys())}, ignore_index=True)
        mean = np.mean(list(value.values()))
        for si in keyword_dict.loc[keyword_dict['category'] == key, 'keyword'].values:
            print(si)
            for i in list(si):
                if i not in value.keys():
                    value[i] = mean
        value = dict(sorted(value.items(), key=lambda x: (float(x[1])), reverse=True))
        new_value = {}
        keys = value.keys()
        for k in list(keys)[0:int(0.8*len(keys))]:
            new_value[k] = value[k]
        to_dict[key] = new_value
    print(to_dict)


def out_key_word():
    print("result")


def forecast_results(X, y):
    c_nb = ComplementNB()
    transfer = TfidfTransformer()
    X = transfer.fit_transform(X)
    c_nb.fit(X, y)
    print(c_nb.predict(X))
    print("准确率为:", c_nb.score(X, y))
    print(c_nb.predict_proba(X))
