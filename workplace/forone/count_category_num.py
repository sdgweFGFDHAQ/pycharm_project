import pandas as pd
import numpy as np


def count_the_number_of_categories(csv_data):
    word_list = []
    for v in csv_data['word_name']:
        word_list.extend(v)
    print(word_list)
    dummies = pd.DataFrame(np.zeros((len(csv_data), len(word_list))), columns=word_list)
    for index in range(0, len(csv_data)):
        for word in csv_data['word_name'].iloc[index]:
            if word in word_list:
                dummies.loc[index, word] = 1
    # dummies = pd.get_dummies(word_list)
    print('特征词转向量:{}'.format(dummies))
    return dummies


def get_info_gain_rate(dummies, categories):
    # 属性信息熵
    info_gain_list = dict()
    entropy = get_info_entropy(categories)
    for index, row in dummies.items():
        d = dict()
        for i in list(range(len(row))):
            d[row[i]] = d.get(row[i], []) + [categories[i]]
        cond_entropy = sum([get_info_entropy(d[k]) * len(d[k]) / float(len(row)) for k in d])
        info_gain = entropy - cond_entropy
        # 信息增益率
        info_intrinsic = - sum([np.log2(len(d[k]) / float(len(row))) * len(d[k]) / float(len(row)) for k in d])
        info_gain_rate = info_gain / info_intrinsic
        info_gain_list[index] = info_gain_rate
    info_gain_list = sorted(info_gain_list.items(), key=lambda x: x[1], reverse=True)
    # print(info_gain_list)
    pd.DataFrame(info_gain_list).to_csv('../save_info_weight.txt', index=False)
    return info_gain_list


def get_info_gain(dummies, categories):
    # 属性信息熵
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
    print('根据信息增益获取特征词:{}'.format(info_gain_list))
    return info_gain_list


def get_info_entropy(categories):
    # 类别信息熵
    if not isinstance(categories, pd.core.series.Series):
        categories = pd.Series(categories)
    cg_ary = categories.groupby(by=categories).count().values / float(len(categories))
    return -(np.log2(cg_ary) * cg_ary).sum()
