import numpy as np
import jieba.analyse
import pandas as pd
from workplace.forone.train_model import tf_idf_by_python, b_train_parameter
from workplace.forone.count_category_num import count_the_number_of_categories, get_info_gain, get_info_gain_rate
import re


# 获取处理好的数据
def get_data_from_CSV():
    csv_data = pd.read_csv('../guangzhou.csv', usecols=['id', 'name', 'type', 'typecode'], nrows=1000)
    csv_data['word_name'] = csv_data['name'].apply(cut_word)
    csv_data['word_name'].to_csv('../cut_word_list.csv', index=False)
    print(csv_data.head(10))
    return csv_data


# 分词并过滤无用字符
def cut_word(word):
    out_word_list = []
    # 加载停用词
    stop_words = stop_words_list('../stop_word_plug.txt')
    word = re.sub(r'\(.*?\)', '', word)
    word = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '', word)
    # 不可分割的词
    with open('../inseparable_word_list.txt', 'r', encoding='utf-8') as in_word:
        for iw in in_word:
            iw = iw.strip('\n')
            jieba.suggest_freq(iw, True)
    l_cut_words = jieba.lcut(word)
    for lc_word in l_cut_words:
        if lc_word not in stop_words:
            if lc_word != '\t':
                out_word_list.append(lc_word)
    return out_word_list


# 创建停用词list
def stop_words_list(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# jieba实现算法
def find_category0(csv_data):
    name_text = ''.join(csv_data['name'].tolist())
    category = jieba.analyse.extract_tags(name_text, topK=10, withWeight=True, allowPOS=())
    print(category)


# 手写算法
def find_category(csv_data):
    category = tf_idf_by_python(csv_data['word_name'].tolist())
    print(category)
    return category


# 对数据进行统计保存
def save_data_info(csv_data):
    # 对类别分组计数
    type_group = csv_data[['typecode']].groupby(csv_data['type'])
    category_frequency = type_group.value_counts()
    print(category_frequency)
    category_frequency.to_csv('../term_category_frequency.csv', index=True)
    # csv_2.to_csv('../category_frequency.csv', index=False)


if __name__ == '__main__':
    # 获取处理好的数据
    csv_data = get_data_from_CSV()
    # 对数据进行统计保存
    # save_data_info(csv_data)
    # TF-IDF获取高频特征词
    # category = find_category(csv_data)
    # 信息增益获取高频特征词
    dummies = count_the_number_of_categories(csv_data)
    gain_list = get_info_gain_rate(dummies, csv_data['type'])
    # 训练贝叶斯模型
    b_train_parameter(dummies, csv_data['type'])
    # 对新数据分类
    # new_data = csv_data.iloc[4000]
    # print(new_data)
    # b_use_model(new_data)