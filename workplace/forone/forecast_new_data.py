import sys
import time
from ast import literal_eval
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB, BernoulliNB
from workplace.forone.global_parameter import StaticParameter as SP
from workplace.forone.mini_tool import cut_word


# 判断新数据
def classify_forecast_results(names_data):
    model_data = pd.read_csv('../result_model.csv',
                             usecols=['category', 'category_words'])
    classify_result_list = list()
    category_words_list = []
    for category_words in model_data['category_words']:
        category_words_list.append(eval(category_words))
    probability = dict(zip(model_data['category'].values, category_words_list))
    for name_list in names_data['cut_name'].values:
        classify_result = {}
        for word in name_list:
            for category_key in probability.keys():
                classify_result[category_key] = 0
                if word in probability[category_key].keys():
                    classify_result[category_key] += probability[category_key][word]
        sort_result = {}
        sort_result0 = dict(sorted(classify_result.items(), key=lambda x: (float(x[1])), reverse=True))
        keys = sort_result0.keys()
        for k in list(keys)[0:len(keys)]:
            sort_result[k] = sort_result0[k]
        classify_result_list.append(sort_result)
    batch_data_classify = pd.DataFrame({'names': names_data['name'], 'category_result': classify_result_list})
    batch_data_classify.to_csv('../atest.csv', index=False)
    print(batch_data_classify)


def calculate_accuracy(tset_data):
    bayes_results = bayes_forecast_results(test_data)
    real_result = list(tset_data['category3_new'])
    forecast_result = list()
    classify_data = pd.read_csv('../atest.csv')
    categories = classify_data['category_result'].apply(literal_eval)
    for category in categories:
        result_k = list(category.keys())[0]
        if category.get(result_k) == 0 or category.get(result_k) == '0':
            forecast_result.append('无')
        else:
            forecast_result.append(result_k)
    # 计算
    count = 0
    data_num = len(real_result)
    for i in range(data_num):
        if forecast_result[i] == '无':
            forecast_result[i] = bayes_results[i]
        if real_result[i] == forecast_result[i]:
            count += 1
    print("分类模型准确率：", count / data_num)


def bayes_forecast_results(test_data):
    # 训练贝叶斯模型
    train_data = pd.read_csv('../standard_store_gz.csv', usecols=['name', 'category3_new', 'cut_name'], nrows=80000)
    train_data['cut_name'] = train_data['cut_name'].apply(literal_eval)
    cut_name_list = list()
    for i in list(train_data['cut_name']):
        cut_name_list.append(' '.join(i))
    c_v = CountVectorizer()
    train_x = c_v.fit_transform(cut_name_list).astype(np.int8)
    c_nb = ComplementNB()
    # nb2 = MultinomialNB()
    # nb3 = BernoulliNB()
    # for model in [c_nb, nb2, nb3]:
    #     model.fit(X, y)
    #     print(sys.getsizeof(X) / 1024 / 1024, 'MB')
    #     print(model.predict(X))
    #     print("准确率为:", model.score(X, y))
    c_nb.fit(train_x, train_data['category3_new'])
    # 测试
    cut_name_list = list()
    for i in test_data['cut_name']:
        cut_name_list.append(' '.join(i))
    test_x = c_v.transform(cut_name_list).astype(np.int8)
    predict_result = c_nb.predict(test_x)
    print(predict_result)
    pd.DataFrame(predict_result, columns=['category']).to_csv('../predict_result.csv')
    print("准确率为:", c_nb.score(test_x, test_data['category3_new']))
    return list(predict_result)


if __name__ == '__main__':
    print("=======数据分词提取特征=======", time.localtime(time.time()))
    # # 店名分词-无标签的应用数据
    # csv_data = pd.read_csv(SP.TEST_DATA_PATH, usecols=['name'], nrows=100)
    # csv_data = csv_data[csv_data['category1_new'].notnull() & (csv_data['category1_new'] != "")]
    # csv_data['cut_name'] = csv_data['name'].apply(cut_word)
    # csv_data['cut_name'] = csv_data['cut_name'].apply(literal_eval)
    # print(csv_data.head(3))
    # 店名分词-有标签的测试数据
    test_data = pd.read_csv(SP.TEST_DATA_PATH, usecols=['name', 'category3_new'], nrows=100)
    test_data['cut_name'] = test_data['name'].apply(cut_word)
    print(test_data.head(3))
    print("=======结束分词=======", time.localtime(time.time()))
    print("=======开始分类模型分类预测======", time.localtime(time.time()))
    # 计算模型准确率
    classify_forecast_results(test_data)
    calculate_accuracy(test_data)
    print("=======结束预测=======", time.localtime(time.time()))
    # print("=======开始贝叶斯模型分类预测======", time.localtime(time.time()))
    # 比对贝叶斯
    # bayes_forecast_results(test_data)
    # print("=======结束贝叶斯模型分类预测=======", time.localtime(time.time()))
