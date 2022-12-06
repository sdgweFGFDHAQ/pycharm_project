import pandas as pd
import re
import jieba
import time
from workplace.forone.count_category_num import count_the_number_of_categories, get_info_gain_rate, get_categories
from workplace.forone.train_model import b_train_parameter
from ast import literal_eval


def set_file_standard_data(path):
    csv_data = pd.read_csv(path, usecols=['name', 'category1_new', 'category2_new', 'category3_new'])
    # 用一级标签填充空白(NAN)的二级标签、三级标签
    csv_data['category2_new'].fillna(csv_data['category1_new'], inplace=True)
    csv_data['category3_new'].fillna(csv_data['category2_new'], inplace=True)
    # 得到标准数据
    csv_data['cut_name'] = csv_data['name'].apply(cut_word)
    csv_data.to_csv('../standard_store_gz.csv', columns=['name', 'category3_new', 'cut_name'])
    # 各级标签映射字典
    category = csv_data[['category1_new', 'category2_new', 'category3_new']]
    category = category.drop_duplicates(keep='first')
    category.reset_index(inplace=True, drop=True)
    category.to_csv('../category_dict.csv')
    print("类别个数：", len(category['category3_new']))


def cut_word(word):
    out_word_list = []
    # 加载停用词
    stop_words = [line.strip() for line in open('../stop_word_plug.txt', 'r', encoding='utf-8').readlines()]
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


def set_category_words():
    category_keyword = pd.read_csv('../di_keyword_map.csv')
    ck = category_keyword.groupby(by='category')['keyword'].apply(list)
    ck.to_csv('../keyword_dict.csv')
    print("品类种数：", len(ck))


def get_data():
    csv_data = pd.read_csv('../standard_store_gz.csv', usecols=['name', 'category3_new', 'cut_name'], nrows=2000)
    csv_data['cut_name'] = csv_data['cut_name'].apply(literal_eval)
    print(csv_data.head(10))
    return csv_data


if __name__ == '__main__':
    # 前期准备：获取店名数据，统计三级分类
    # data_path = '../di_store_gz.csv'
    # set_file_standard_data(data_path)
    # 前期准备：更新每种类别对应的关键字
    # set_category_words()
    # 先构建一个空间向量再说
    data = get_data()
    dummy = count_the_number_of_categories(data)
    gain_lists = get_info_gain_rate(dummy, data['category3_new'])
    print(gain_lists)
    info_gain_list = dict(sorted(gain_lists.items(), key=lambda x: x[1], reverse=True))
    new_dummies = get_categories(list(info_gain_list.keys()), data)
    b_train_parameter(new_dummies, data['category3_new'])

