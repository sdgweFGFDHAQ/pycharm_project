import sys
import time
from ast import literal_eval
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import ComplementNB, MultinomialNB, BernoulliNB
from workplace.forone.global_parameter import StaticParameter as SP
from workplace.forone.mini_tool import cut_word


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


def forecast_results(csv_data):
    c_nb = ComplementNB()
    # nb2 = MultinomialNB()
    # nb3 = BernoulliNB()
    transfer = TfidfTransformer()
    X = transfer.fit_transform(csv_data['cut_name'])
    # for model in [c_nb, nb2, nb3]:
    #     model.fit(X, y)
    #     print(sys.getsizeof(X) / 1024 / 1024, 'MB')
    #     print(model.predict(X))
    #     print("准确率为:", model.score(X, y))
    c_nb.fit(X, y)
    print(c_nb.predict(X))
    print("准确率为:", c_nb.score(X, y))
    print(c_nb.predict_proba(X))


def new_forecast_results(x, y):
    count = 0
    csv = pd.read_csv('../atest.csv')
    category = list(csv['category'])
    for i in range(len(y)):
        if y[i] == category[i]:
            count += 1
    return count / len(y)


if __name__ == '__main__':
    print("=======数据分词提取特征=======", time.localtime(time.time()))
    # 店名分词
    csv_data = pd.read_csv(SP.TEST_DATA_PATH, usecols=['name'], nrows=100)
    csv_data['cut_name'] = csv_data['name'].apply(cut_word)
    csv_data['cut_name'] = csv_data['cut_name'].apply(literal_eval)
    print(csv_data.head(3))
    print("=======结束分词=======", time.localtime(time.time()))
    # 获得分类结果
    calculate_category(csv_data)
    # 计算模型准确率
    new_forecast_results(csv_data['name'], csv_data['category3_new'])
    print("=======结束预测=======", time.localtime(time.time()))
    # 比对贝叶斯
    forecast_results(csv_data)
    print("=======结束贝叶斯分类=======", time.localtime(time.time()))