import pandas as pd
import json
import xlrd
from icecream import ic


def merge_multiple_categories(source_path, target_path, use_cols, keep_cols, merge_col):
    csv_data = pd.read_csv(source_path, usecols=use_cols, keep_default_na=False)
    csv_data.drop_duplicates(keep='first', inplace=True)
    print("非单源高德 原始数据集：", csv_data.shape[0])

    # 修改列名
    csv_data['storeType'] = csv_data['category1_new']
    save_cols = keep_cols + [merge_col]

    # 用一级标签填充空白(NAN)的二级标签、三级标签
    # 删除至少有3个NaN值的行 # data = data.dropna(axis=0, thresh=3)
    # csv_data['category2_new'].fillna(csv_data['category1_new'], inplace=True)
    # csv_data['storeType'].fillna(csv_data['category2_new'], inplace=True)
    # csv_data['storeType'].fillna(csv_data['predict_category'], inplace=True)
    # csv_data = csv_data[csv_data['storeType'].notnull() & (csv_data['storeType'] != '')]

    csv_result = csv_data[save_cols]
    csv_result.drop_duplicates(keep='first', inplace=True)
    print("去重数据集：", csv_result.shape[0])
    csv_result = csv_result.set_index(keep_cols)

    # 合并售卖的商品
    grouped = csv_result.groupby(by=keep_cols)
    result = grouped.agg({merge_col: lambda x: set(x)})
    result = result.reset_index()
    print("合并饮料商品：", result.shape[0])
    result.to_csv(target_path, index=False, columns=save_cols)


def process_data(source_path, target_path, keep_cols, merge_col):
    exist_df = pd.read_csv(source_path, usecols=keep_cols + [merge_col], keep_default_na=False)
    print('csv文件数据量:', exist_df.shape[0])
    # exist_df['storeType'].fillna(exist_df['predict_category'], inplace=True)
    exist_df = exist_df[exist_df['storeType'].notnull() & (exist_df['storeType'] != '')].reset_index(drop=True)
    print('存在店铺，有所售商品类别的数据量：{}条！'.format(exist_df.shape[0]))

    # 将'drinkTypes'列的列表元素提取为新的列
    new_columns = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
                   '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']

    extracted_columns = exist_df[merge_col].apply(
        lambda x: [1 if item in x else 0 for item in new_columns]).tolist()
    extracted_df = pd.DataFrame(extracted_columns, columns=new_columns, dtype=int).reset_index(drop=True)
    # 将提取的列添加到原始DataFrame中
    for c in new_columns:
        exist_df[c] = extracted_df[c]
    exist_df['labels_token'] = extracted_columns

    exist_df.to_csv(target_path, index=False, encoding='utf-8')
    # 统计新增列中值为1的数量
    column_counts = exist_df[new_columns].sum()
    print(column_counts)


if __name__ == '__main__':
    read_path = 'C:\\Users\\86158\\Desktop\\export_202307261353.csv'
    save_path_temp = 'C:\\Users\\86158\\Desktop\\di_sku_log_data_temp.csv'
    save_path = 'C:\\Users\\86158\\Desktop\\di_sku_log_chain_data.csv'
    use_columns = ['store_id', 'name', 'category1_new', 'predict_category', 'drink_label']
    keep_columns = ['store_id', 'name', 'storeType', 'predict_category']
    merge_column = 'drink_label'

    # 合并同store_id数据，调整部分字段
    # merge_multiple_categories(read_path, save_path_temp, use_columns, keep_columns, merge_column)

    # 提取0-1饮料标签
    process_data(save_path_temp, save_path, keep_columns, merge_column)
