import numpy as np
import pandas as pd


# 原始向量空间
def count_the_number_of_categories(csv_data):
    word_list = []
    for v in csv_data['cut_name']:
        word_list.extend(v)
    new_word_list = list(set(word_list))
    # new_word_list.sort(key=word_list.index)
    print(new_word_list[:10])
    dummies = pd.DataFrame(np.zeros((len(csv_data), len(new_word_list)), dtype=np.int8), columns=new_word_list)
    for index in range(0, len(csv_data)):
        for word in csv_data['cut_name'].iloc[index]:
            if word in new_word_list:
                dummies.loc[index, word] += 1
    # dummies = pd.get_dummies(word_list)
    print('特征词转向量:{}'.format(dummies.head(3)))
    return dummies
    # return dummies.astype('category')


# 提取高频词的向量空间
def get_categories(word_list_key, csv_data):
    word_list = word_list_key[: int(0.8 * len(word_list_key))]
    print(word_list)
    dummies = pd.DataFrame(np.zeros((len(csv_data), len(word_list))), columns=word_list)
    for index in range(0, len(csv_data)):
        for word in csv_data['cut_name'].iloc[index]:
            if word in word_list:
                dummies.loc[index, word] = 1
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
    return info_gain_list


def get_info_gain(dummies, categories, gain_lists):
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
    gain_lists.update(info_gain_list)
    return info_gain_list


def get_info_entropy(categories):
    # 类别信息熵
    if not isinstance(categories, pd.core.series.Series):
        categories = pd.Series(categories)
    cg_ary = categories.groupby(by=categories).count().values / float(len(categories))
    return -(np.log2(cg_ary) * cg_ary).sum()
