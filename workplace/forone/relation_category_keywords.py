import ast
import re
import numpy as np
import pandas as pd
from multiprocessing import Manager, Pool
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB, BernoulliNB


# 获得特征词权重
def get_feature_prob(X, y):
    result_dict = calculate_feature_prob(X, y)
    df = pd.DataFrame({'category': result_dict.keys(), 'keyword': result_dict.values()})
    print(df.head(5))
    return result_dict


# 得到特征词权重并过滤无关特征
def calculate_feature_prob(X, y):
    c_nb = ComplementNB()
    c_nb.fit(X, y)
    c_nb.feature_log_prob_ = -c_nb.feature_log_prob_
    feature_prob = pd.DataFrame(np.float32(np.exp(c_nb.feature_log_prob_)), index=c_nb.classes_, columns=X.columns)
    # feature_prob = pd.DataFrame(np.float32(c_nb.feature_log_prob_), index=c_nb.classes_, columns=X.columns)
    # 获取根据贝叶斯计算的权重
    to_dict = feature_prob.to_dict(orient='index')
    # 只获取该类别下出现过的关键词
    cut_name_dict = pd.read_csv('../cut_name_dict.csv')
    # 获取每个类别对应的权重
    result_dict = dict()
    for index, row in cut_name_dict.iterrows():
        cnd_category = row['category3_new']
        cnd_cut_name = ast.literal_eval(row['cut_name'])
        word_weight = dict()
        if cnd_category not in to_dict.keys():
            continue
        kv_weight = to_dict[cnd_category]
        # 把to_dict里对应的权重赋给cut_name_dict
        for kw in cnd_cut_name:
            if kw not in kv_weight.keys():
                continue
            word_weight[kw] = kv_weight[kw]
        kd_keyword = re.sub(r"\[|\]|'|\"", '', ndarray_values)
        kd_keyword = kd_keyword.replace(' ', '').split(',')
        for i_key in kd_keyword:
            if i_key not in word_weight:
                continue
            if i_key not in cnd_cut_name:
                word_weight[i_key] = mean
            else:
                word_weight[i_key] += 1
        desc_word_weight = dict(sorted(word_weight.items(), key=lambda x: (float(x[1])), reverse=True))
        result_dict[cnd_category] = desc_word_weight
    print(result_dict)
    return result_dict


# 获得特征词权重-多进程
def get_feature_prob_part(X, y):
    # 管理全局参数
    result_dict = Manager().dict()
    # 只获取该类别下出现过的关键词
    cut_name_dict = pd.read_csv('../cut_name_dict.csv')
    # 获取预设的关键词
    keyword_dict = pd.read_csv('../keyword_dict.csv')
    # 调用线程池
    pool = Pool(processes=6)
    c_num = len(X.columns)
    dummy = [int(c_num * i / 6) for i in range(7)]
    for i in range(6):
        dummy_i = X.iloc[:, dummy[i]:dummy[i + 1]]
        pool.apply_async(calculate_feature_prob_part, args=(dummy_i, y, result_dict, cut_name_dict, keyword_dict))
    pool.close()
    pool.join()
    df = pd.DataFrame({'category': result_dict.keys(), 'keyword': result_dict.values()})
    print(df)


# 计算特征词权重-多进程
def calculate_feature_prob_part(dummy_i, y, result_dict, cut_name_dict, keyword_dict):
    c_nb = ComplementNB()
    c_nb.fit(dummy_i, y)
    feature_prob = pd.DataFrame(c_nb.feature_log_prob_, index=c_nb.classes_, columns=dummy_i.columns)
    # 获取根据贝叶斯计算的权重
    to_dict = feature_prob.to_dict(orient='index')
    # 获取每个特征对应的权重
    dummy_to_cate = dict()
    for di in dummy_i.columns:
        dummy_to_cate[di] = dict(feature_prob[di])
    for index, row in cut_name_dict.iterrows():
        cnd_category = row['category3_new']
        cnd_cut_name = ast.literal_eval(row['cut_name'])
        word_weight = dict()
        if cnd_category not in to_dict.keys():
            continue
        kv_weight = to_dict[cnd_category]
        # 把to_dict里对应的权重赋给cut_name_dict
        for kw in cnd_cut_name:
            if kw not in kv_weight.keys():
                continue
            word_weight[kw] = kv_weight[kw]
        # 把keyword_dict的关键词，加上平均值赋给cut_name_dict
        mean = np.mean(list(word_weight.values()))
        ndarray_values = str(keyword_dict.loc[keyword_dict['category'] == cnd_category, 'keyword'].values)
        kd_keyword = re.sub(r"\[|\]|'|\"", '', ndarray_values)
        kd_keyword = kd_keyword.replace(' ', '').split(',')
        for i_key in kd_keyword:
            if i_key not in word_weight:
                continue
            if i_key not in cnd_cut_name:
                word_weight[i_key] = mean
            else:
                word_weight[i_key] += 1
        desc_word_weight = dict(sorted(word_weight.items(), key=lambda x: (float(x[1])), reverse=True))
        result_dict[cnd_category] = desc_word_weight
    return result_dict


# 加入人工设置的特征词
def add_artificial_keywords(category_dict):
    # 获取预设的关键词
    keyword_dict = pd.read_csv('../keyword_dict.csv')
    for index, row in category_dict.iterrows():
        # 把keyword_dict的关键词，加上平均值赋给cut_name_dict
        mean = np.mean(list(row.values()))
        ndarray_values = str(keyword_dict.loc[keyword_dict['category'] == cnd_category, 'keyword'].values)


# 输出指定格式的模型,带权重
def out_keyword(prob):
    core_words = []
    category_words = []
    for category, words in prob.items():
        key_word = words.keys()
        core_word = []
        category_word = dict()
        # 对特征词切片，区分品类词和核心词
        if len(key_word) < 50:
            category_word = words
        else:
            for k in list(key_word)[:50]:
                category_word[k] = words[k]
            for k in list(key_word)[50:]:
                core_word.append(k)
        category_words.append(category_word)
        core_words.append(core_word)
    result_model = pd.DataFrame(
        {'category': prob.keys(), 'category_words': category_words, 'core_words': core_words})
    result_model.to_csv('../result_model.csv', index=False)
    print(result_model.head(10))
    return result_model
# float_format='%.3f'
