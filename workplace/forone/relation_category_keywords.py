import ast
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB, BernoulliNB
import sys


# 获得特征词权重
def get_feature_prob(X, y):
    c_nb = ComplementNB()
    c_nb.fit(X, y)
    feature_prob = pd.DataFrame(c_nb.feature_log_prob_, index=c_nb.classes_, columns=X.columns)
    # 获取根据贝叶斯计算的权重
    to_dict = feature_prob.to_dict(orient='index')
    # 只获取该类别下出现过的关键词
    cut_name_dict = pd.read_csv('../cut_name_dict.csv')
    # 获取预设的关键词
    keyword_dict = pd.read_csv('../keyword_dict.csv')
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
        # 把keyword_dict的关键词，加上平均值赋给cut_name_dict
        mean = np.mean(list(word_weight.values()))
        ndarray_values = str(keyword_dict.loc[keyword_dict['category'] == cnd_category, 'keyword'].values)
        kd_keyword = re.sub(r"\[|\]|'|\"", '', ndarray_values).split(',')
        for i_key in kd_keyword:
            if i_key not in cnd_cut_name:
                word_weight[i_key] = mean
            word_weight[i_key] = mean + 0.1
        result_dict[cnd_category] = word_weight
    df = pd.DataFrame({'category': result_dict.keys(), 'keyword': result_dict.values()})
    df.to_csv('../filename.csv', index=False)
    return result_dict


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


# 输出指定格式的模型,带权重
def out_keyword(prob):
    core_words = []
    category_words = []
    for key, value in prob.items():
        keys = value.keys()
        core_word = []
        category_word = {}
        for k in list(keys)[0:int(0.03 * len(keys))]:
            category_word[k] = value[k]
        for k in list(keys)[int(0.03 * len(keys)):]:
            core_word.append(k)
        category_words.append(category_word)
        core_words.append(core_word)
    result_model = pd.DataFrame(
        {'category': prob.keys(), 'category_words': category_words, 'core_words': core_words})
    result_model.to_csv('../result_model.csv', index=False)


# 输出指定格式的模型,不带权重
def out_keyword_no_weight(to_dict):
    core_words = []
    category_words = []
    for key, value in to_dict.items():
        keys = value.keys()
        core_word = []
        category_word = []
        for k in list(keys)[0:int(0.1 * len(keys))]:
            category_word.append(k)
        for k in list(keys)[int(0.1 * len(keys)):int(0.3 * len(keys))]:
            core_word.append(k)
        category_words.append(category_word)
        core_words.append(core_word)
    result_model = pd.DataFrame(
        {'category': to_dict.keys(), 'category_words': category_words, 'core_words': core_words})
    result_model.to_csv('../result_model_no_weight.csv', index=False)


# 判断新数据
def calculate_category(names):
    categories = []
    model_data = pd.read_csv('../result_model.csv',
                             usecols=['category', 'category_words'])
    for name_list in names['cut_name'].values:
        category = judge_category(name_list, model_data)
        categories.append(category)
    df = pd.DataFrame({'names': names['name'], 'category': categories})
    df.to_csv('../atest.csv', index=False)


def judge_category(name_list, model_data):
    sort_result = {}
    probability = dict(zip(model_data['category'].tolist(), np.zeros((len(model_data['category'])))))
    for word in name_list:
        for index, row in model_data.iterrows():
            category_words_dict = ast.literal_eval(row['category_words'])
            if word in category_words_dict:
                probability[row['category']] += category_words_dict[word]
        sort_result0 = dict(sorted(probability.items(), key=lambda x: (float(x[1])), reverse=True))
        keys = sort_result0.keys()
        for k in list(keys)[0:int(0.1 * len(keys))]:
            sort_result[k] = sort_result0[k]
    return sort_result


def forecast_results(X, y):
    c_nb = ComplementNB()
    # nb2 = MultinomialNB()
    # nb3 = BernoulliNB()
    transfer = TfidfTransformer()
    X = transfer.fit_transform(X)
    print(sys.getsizeof(X) / 1024 / 1024, 'MB')
    # for model in [c_nb, nb2, nb3]:
    #     model.fit(X, y)
    #     print(model.predict(X))
    #     print("准确率为:", model.score(X, y))
    c_nb.fit(X, y)
    print(c_nb.predict(X))
    print("准确率为:", c_nb.score(X, y))
    # print(c_nb.predict_proba(X))


def new_forecast_results(x, y):
    count = 0
    csv = pd.read_csv('../atest.csv')
    category = list(csv['category'])
    for i in range(len(y)):
        if y[i] == category[i]:
            count += 1
    return count / len(y)
