import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from workplace.forone.global_parameter import StaticParameter as SP


# 特征向量化
def feature_vectorization(csv_data):
    # 每个样本特征的集合
    cut_name_list = list()
    for i in csv_data['cut_name']:
        cut_name_list.append(' '.join(i))
    # 对分类无帮助的特征集合
    useless_feature = []
    # 获取稀疏矩阵
    c_v = CountVectorizer()
    vector_matrix = c_v.fit_transform(cut_name_list).astype(np.int8).toarray()
    sparse_matrix = csc_matrix(vector_matrix)
    # 列特征集合
    feature_list = c_v.get_feature_names_out()
    # 特征在店铺样本中出现的样本数
    nonzero_column_list = np.diff(sparse_matrix.indptr)
    # 去除在少于10种或多于75%的商品中出现的词
    for index, feature in enumerate(feature_list):
        num = nonzero_column_list[index]
        if num < SP.MIN_NUMBER or num > len(vector_matrix) * SP.MAX_RATE:
            useless_feature.append(feature)
    dummy = pd.DataFrame(vector_matrix, index=csv_data['name'], columns=feature_list)
    dummy.drop(useless_feature, axis=1, inplace=True)
    return dummy, c_v
    # return dummies.astype('category')


def reduce_by_mutual(dummy, category):
    igr_list = dict(zip(dummy.columns, mutual_info_classif(dummy, category, discrete_features=True)))
    low_igr_feature = list()
    igr_dict = dict(sorted(igr_list.items(), key=lambda x: (float(x[1])), reverse=False))
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
