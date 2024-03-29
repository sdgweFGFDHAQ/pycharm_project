import ast
import re
import time
import numpy as np
import pandas as pd
from multiprocessing import Manager, Pool
from sklearn import preprocessing
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB, BernoulliNB
from workplace.label_nb.mini_tool import error_callback
from workplace.label_nb.global_parameter import StaticParameter as SP


# 获得特征词权重-多进程
def get_feature_prob_part(X, y) -> dict:
    # 只获取该类别下出现过的关键词
    cut_name_dict = pd.read_csv('../dict_cutname.csv')
    # 调用线程池
    pool = Pool(processes=4)
    result_dict = Manager().dict()
    for cate in y.values:
        result_dict[cate] = dict()
    c_num = len(X.columns)
    dummy_index = [int(c_num * i / 4) for i in range(5)]
    for i in range(4):
        dummy_i = X.iloc[:, dummy_index[i]:dummy_index[i + 1]]
        pool.apply_async(calculate_feature_prob_part, args=(dummy_i, y, cut_name_dict, result_dict, i),
                         error_callback=error_callback)
    pool.close()
    pool.join()
    return result_dict


# 计算特征词权重-多进程
def calculate_feature_prob_part(dummy_i, y, cut_name_dict, result_dict, i):
    c_nb = ComplementNB()
    c_nb.fit(dummy_i, y)
    # 把取feature_log_prob_值
    feature_prob = pd.DataFrame(np.float32(c_nb.feature_log_prob_), index=c_nb.classes_,
                                columns=dummy_i.columns)
    # 获取根据贝叶斯计算的权重
    to_dict = feature_prob.to_dict(orient='index')
    # 获取每个类别对应的权重
    for index, row in cut_name_dict.iterrows():
        word_weight = dict()
        cnd_category = row['category3_new']
        cnd_cut_name = ast.literal_eval(row['cut_name'])
        if cnd_category not in to_dict.keys():
            continue
        for feature in cnd_cut_name:
            if feature in to_dict[cnd_category].keys():
                word_weight[feature] = to_dict[cnd_category][feature]
        new_result_dict = result_dict[cnd_category] | word_weight
        result_dict[cnd_category] = new_result_dict
    print(result_dict)


# 加入人工设置的特征词
def add_artificial_keywords(result_dict):
    # 获取预设的关键词
    feature_weight_dict = result_dict.copy()
    keyword_dict = pd.read_csv('../dict_keyword.csv')
    keyword_dict = dict(zip(keyword_dict['category'], keyword_dict['keyword']))
    for key in keyword_dict.keys():
        # 把keyword_dict的关键词，加上平均值赋给cut_name_dict
        word_weight = dict()
        if key not in result_dict:
            continue
        if len(result_dict[key].keys()) == 0:
            mean = 0
        else:
            mean = np.mean(list(result_dict[key].values()))
        keyword_list = keyword_dict[key]
        kd_keyword = re.sub(r"\[|\]|'|\"", '', keyword_list)
        kd_keyword = kd_keyword.replace(' ', '').split(',')
        for i_key in kd_keyword:
            if i_key not in result_dict[key].keys():
                word_weight[i_key] = mean
            else:
                word_weight[i_key] = result_dict[key][i_key] * 1.5
        feature_weight_dict[key].update(word_weight)
        feature_weight_dict[key] = dict(
            sorted(feature_weight_dict[key].items(), key=lambda x: (float(x[1])), reverse=True))
    return feature_weight_dict


# 输出指定格式的模型,带权重
def out_keyword(prob):
    core_words = list()
    category_words = list()
    for category, words in prob.items():
        key_word = words.keys()
        core_word = list()
        category_word = dict()
        # 对特征词切片，区分品类词和核心词
        if len(key_word) < SP.CATEGORY_WORDS_NUM:
            category_word = words
        else:
            for k in list(key_word)[:SP.CATEGORY_WORDS_NUM]:
                category_word[k] = words[k]
            for k in list(key_word)[SP.CATEGORY_WORDS_NUM:]:
                core_word.append(k)
        category_word_percent = preprocessing.minmax_scale(list(category_word.values()))
        category_word = dict(zip(category_word.keys(), category_word_percent))
        category_words.append(category_word)
        core_words.append(core_word)
    result_model = pd.DataFrame(
        {'category': prob.keys(), 'category_words': category_words, 'core_words': core_words})
    result_model.to_csv('../result_model.csv', index=False)
    print(result_model.head(5))
    return result_model
# float_format='%.3f'
