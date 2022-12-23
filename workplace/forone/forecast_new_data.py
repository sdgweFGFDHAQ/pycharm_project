import ast
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import ComplementNB


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
