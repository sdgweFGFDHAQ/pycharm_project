import time
from ast import literal_eval
import pandas as pd
import numpy as np
from workplace.forone.mini_tool import cut_word
from workplace.forone.count_category_num import feature_vectorization, reduce_by_mutual
from workplace.forone.relation_category_keywords import get_feature_prob, get_feature_prob_part, out_keyword
from workplace.forone.forecast_new_data import forecast_results, calculate_category, new_forecast_results


# 读取原始文件,将数据格式标准化
def set_file_standard_data(path):
    csv_data = pd.read_csv(path, usecols=['name', 'category1_new', 'category2_new', 'category3_new'])
    # 用一级标签填充空白(NAN)的二级标签、三级标签
    csv_data['category2_new'].fillna(csv_data['category1_new'], inplace=True)
    csv_data['category3_new'].fillna(csv_data['category2_new'], inplace=True)
    # 得到各级标签映射字典
    category = csv_data[['category1_new', 'category2_new', 'category3_new']]
    category = category.drop_duplicates(keep='first')
    category.reset_index(inplace=True, drop=True)
    category.to_csv('../category_dict.csv')
    print("类别个数：", len(category['category3_new']))
    # 得到标准数据
    csv_data['cut_name'] = csv_data['name'].apply(cut_word)
    csv_data.to_csv('../standard_store_gz.csv', columns=['name', 'category3_new', 'cut_name'])
    # 读取分类标准, 设置每个类别对应的关键字
    category_cut_name = csv_data[['category3_new', 'cut_name']]
    # 相比numpy().append(), concatenate()效率更高，适合大规模的数据拼接
    ccn = category_cut_name.groupby(by='category3_new')['cut_name']\
        .apply(lambda x: list(np.unique(np.concatenate(list(x)))))
    cnd = pd.DataFrame({'category3_new': ccn.index, 'cut_name': ccn.values})
    cnd.to_csv('../cut_name_dict.csv')


def set_category_words():
    category_keyword = pd.read_csv('../di_keyword_map.csv')
    ck = category_keyword.groupby(by='category')['keyword'].apply(list)
    print(ck)
    ck.to_csv('../keyword_dict.csv')
    print("品类种数：", len(ck))


def get_data():
    csv_data = pd.read_csv('../standard_store_gz.csv', usecols=['name', 'category3_new', 'cut_name'], nrows=30000)
    csv_data['cut_name'] = csv_data['cut_name'].apply(literal_eval)
    print(csv_data.head(3))
    return csv_data


if __name__ == '__main__':
    # 前期准备：获取店名数据，统计三级分类
    # set_file_standard_data(SP.DATA_PATH)
    # 前期准备：人为设置每种类别的关键字
    # set_category_words()
    # 获取标准数据
    data = get_data()
    print("=========开始处理数据========", time.localtime(time.time()))
    # 构建一个向量空间
    dummy = feature_vectorization(data)
    # 计算信息增益降维
    new_dummy = reduce_by_mutual(dummy, data['category3_new'])
    print("=======结束构建空间向量=======", time.localtime(time.time()))
    # 获取权重
    prob = get_feature_prob(new_dummy, data['category3_new'])
    print("=========结束权重计算========", time.localtime(time.time()))
    # 输出指定格式的模型
    result_model = out_keyword(prob)
    # 计算模型准确率
    # forecast_results(new_dummy, data['category3_new'])
    # d_f = data.sample(n=100, random_state=111, axis=0)
    # d_f['cut_name'] = d_f['name'].apply(cut_word)
    # calculate_category(d_f)
    # new_forecast_results(d_f['name'], d_f['category3_new'])
    # print("=======代码结束=======", time.localtime(time.time()))
