import pandas as pd
import sklearn
import numpy as np


def count_the_number_of_categories(csv_data):
    word_list = []
    for v in csv_data["word_name"]:
        word_list.extend(v)
    print(word_list)
    dummies = pd.DataFrame(np.zeros((len(csv_data), len(word_list))), columns=word_list)
    for index in range(0, len(csv_data)):
        for word in csv_data["word_name"].iloc[index]:
            if word in word_list:
                dummies[word].iloc[index] = 1
    # dummies = pd.get_dummies(word_list)
    print(dummies)
    # 获取高频特征词
    gain_list = get_info_gain(dummies, csv_data["type"])
    return gain_list


def get_info_gain(dummies, categories):
    info_gain_list = dict()
    entropy = get_info_entropy(categories)
    for index, row in dummies.items():
        d = dict()
        for i in list(range(len(row))):
            d[row[i]] = d.get(row[i], []) + [categories[i]]
        cond_entropy = sum([get_info_entropy(d[k]) * len(d[k]) / float(len(row)) for k in d])
        info_gain = entropy - cond_entropy
        info_gain_list[index] = info_gain
    info_gain_list = sorted(info_gain_list.items(), key=lambda x: x[1], reverse=True)
    print(info_gain_list)
    return info_gain_list


def get_info_entropy(categories):
    # 对类别计数
    if not isinstance(categories, pd.core.series.Series):
        categories = pd.Series(categories)
    cg_ary = categories.groupby(by=categories).count().values / float(len(categories))
    return -(np.log2(cg_ary) * cg_ary).sum()
