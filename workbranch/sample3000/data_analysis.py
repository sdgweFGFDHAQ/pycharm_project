import pandas as pd
import json
import xlrd
from icecream import ic


# 提取json字段，附在原始数据后
def extract_data(*args):
    df = pd.read_excel('../data_score/store_visit_report.xls')
    print(df.columns)
    df_t = df[df['is_exist'] == 't'].copy()
    column_dict = {}
    for col in args:
        column_dict[col] = list()
    df_t["report_json"].apply(getname, args=(column_dict,))
    for col in args:
        df_t[col] = column_dict[col]
    concat = pd.concat((df_t, df[df['is_exist'] == 'f']), axis=0)
    concat.to_csv('new_sv_report.csv')


def getname(df, col_dict):
    value = json.loads(df)
    for k, v in col_dict.items():
        v.append(value[k])


# 处理json字段，抽出分类所需的特征，生成新的csv文件
def get_source_data(column_list):
    df = pd.read_excel('../data_score/store_visit_report.xls')
    print('线下跑店获取的字段列名：'.format(df.columns))
    df_t = df[df['is_exist'] == 't'].copy()

    column_dict = {}
    for col in column_list:
        column_dict[col] = list()
    df_t["report_json"].apply(getname, args=(column_dict,))

    # 考虑加入store_id
    column_dict['store_id'] = list(df_t['store_id'].values)

    result_df = pd.DataFrame(column_dict)
    return result_df


def process_data(required_column_list):
    exist_df = get_source_data(required_column_list)
    # exist_df = exist_df[['name', 'storeType', 'iceBoxState', 'drinkTypes']]
    exist_df = exist_df[['store_id', 'name', 'storeType', 'iceBoxState', 'drinkTypes']]

    num = exist_df[exist_df['drinkTypes'] != '[]'].shape[0]
    print('存在店铺，有所售商品类别的数据量：{}条！'.format(num))

    # 将'drinkTypes'列的列表元素提取为新的列
    new_columns = ['碳酸饮料', '果汁', '茶饮', '水', '乳制品', '植物蛋白饮料', '功能饮料']
    extracted_columns = exist_df['drinkTypes'].apply(lambda x: [1 if item in x else 0 for item in new_columns])
    extracted_df = pd.DataFrame(extracted_columns.tolist(), columns=new_columns)

    # 将提取的列添加到原始DataFrame中
    df_with_extracted_columns = pd.concat([exist_df, extracted_df], axis=1)
    df_with_extracted_columns.to_csv('../../workplace/sv_report_data.csv', index=False)
    # 统计新增列中值为1的数量
    column_counts = df_with_extracted_columns[new_columns].sum()
    print(column_counts)


if __name__ == '__main__':
    # 用于原型网络的input
    # get_source_data('name', 'storeType', 'stackingState', 'iceBoxState', 'drinkTypes')
    # 处理 '卖什么商品' 字段
    process_data(['name', 'storeType', 'stackingState', 'iceBoxState', 'drinkTypes'])
    # extract_data('name', 'storeType', 'stackingState', 'iceBoxState', 'drinkTypes')
'''
存在店铺的数据量：1734条！有所售商品类别的数据量:1399条！
碳酸饮料:1307
果汁:1094
茶饮:995
水:1339
乳制品:965
植物蛋白饮料:743
功能饮料:1008
'''