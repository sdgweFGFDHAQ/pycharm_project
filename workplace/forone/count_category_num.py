import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from workplace.forone.global_parameter import StaticParameter as SP


# 特征向量化
def feature_vectorization(csv_data):
    cut_name_list = list()
    for i in csv_data['cut_name']:
        cut_name_list.append(' '.join(i))
    c_v = CountVectorizer()
    transform = c_v.fit_transform(cut_name_list)
    vector_matrix = transform.toarray().astype(np.int8)
    dummy = pd.DataFrame(vector_matrix, index=csv_data['name'], columns=c_v.get_feature_names_out())
    # 提取高频词的向量空间
    dummy_sum = dummy.sum()
    useless_feature = []
    for index, num in dummy_sum.items():
        if num < SP.MIN_NUMBER or num > dummy_sum.sum() * SP.MAX_RATE:
            useless_feature.append(index)
    dummy.drop(useless_feature, axis=1, inplace=True)
    return dummy
    # return dummies.astype('category')


def reduce_by_IGR(dummy, category):
    igr_list = dict(zip(dummy.columns, mutual_info_classif(dummy, category, discrete_features=True)))
    low_igr_feature = list()
    igr_dict = dict(sorted(igr_list.items(), key=lambda x: (float(x[1])), reverse=False))
    print(igr_dict)
    for k in list(igr_dict.keys())[:int(SP.LOW_IGR_PERCENT * len(igr_dict))]:
        low_igr_feature.append(k)
    dummy.drop(low_igr_feature, axis=1, inplace=True)
    print('去除低信息增益的特征:{}'.format(dummy.head(5)))
    return dummy


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
