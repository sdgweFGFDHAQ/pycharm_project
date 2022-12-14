import ast
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from workplace.forone.tools import cut_word


# 获得特征词权重
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
    return to_dict


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


# 输出指定格式的模型
def out_keyword(csv_data, to_dict):
    core_words = []
    category_words = []
    values = to_dict.values()
    for key, value in to_dict.items():
        keys = value.keys()
        core_word = {}
        category_word = {}
        for k in list(keys)[0:int(0.4 * len(keys))]:
            category_word[k] = value[k]
        for k in list(keys)[int(0.4 * len(keys)):]:
            core_word[k] = value[k]
        category_words.append(category_word)
        core_words.append(core_word)
        result_model = pd.DataFrame({'category': csv_data, 'category_words': category_words, 'core_words': core_words})
        result_model.to_csv('E:\\testwhat\pyProjects\\testPY\\workplace\\result_model.csv', index=False)


# 判断新数据
def coculate_category(names):
    model_data = pd.read_csv('E:\\testwhat\pyProjects\\testPY\\workplace\\result_model.csv', usecols=['category', 'category_words'])
    categories = names.apply(judge_category, args=(model_data))
    df = pd.DataFrame({'names': names, 'category': categories})
    df.to_csv('E:\\testwhat\pyProjects\\testPY\\workplace\\atest.csv', index=False)


def judge_category(name, model_data):
    word_list = cut_word(name)
    probability = dict(zip(model_data['category'].tolist(), np.zeros((len(model_data['category'])))))
    result = []
    for word in word_list:
        for index, row in model_data.iterrows():
            if word.__contains__(row['category_words']):
                probability[row['category']] = ast.literal_eval(row['category_words'])[word]
        sort_result0 = dict(sorted(probability.items(), key=lambda x: (float(x[1])), reverse=True))
        sort_result = {}
        keys = sort_result0.keys()
        for k in list(keys)[0:int(0.1 * len(keys))]:
            sort_result[k] = sort_result0[k]
        result.append(sort_result)
    return result

def forecast_results(X, y):
    c_nb = ComplementNB()
    transfer = TfidfTransformer()
    X = transfer.fit_transform(X)
    c_nb.fit(X, y)
    print(c_nb.predict(X))
    print("准确率为:", c_nb.score(X, y))
    print(c_nb.predict_proba(X))
