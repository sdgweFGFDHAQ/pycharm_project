import pandas as pd
import json
import xlrd
from icecream import ic


def merge_multiple_categories(source_path, target_path, use_columns):
    csv_data = pd.read_csv(source_path, usecols=use_columns)
    csv_data.drop_duplicates(keep='first', inplace=True)
    print(csv_data.shape[0])

    csv_data['store_category'] = csv_data['category3_new']

    # 用一级标签填充空白(NAN)的二级标签、三级标签
    # 删除至少有3个NaN值的行 # data = data.dropna(axis=0, thresh=3)
    csv_data['category2_new'].fillna(csv_data['category1_new'], inplace=True)
    csv_data['store_category'].fillna(csv_data['category2_new'], inplace=True)
    csv_data['store_category'].fillna(csv_data['predict_category'], inplace=True)

    csv_data.to_csv(target_path, columns=use_columns + ['store_category'])


def process_data(source_path, target_path, required_column_list):
    exist_df = pd.read_csv(source_path, usecols=required_column_list, keep_default_na=False)
    print('csv文件数据量:', exist_df.shape[0])
    exist_df = exist_df.fillna('')
    # exist_df = exist_df[exist_df['drink_labels'].notnull() & (exist_df['drink_labels'] != '')]
    print('存在店铺，有所售商品类别的数据量：{}条！'.format(exist_df.shape[0]))

    # 将'drinkTypes'列的列表元素提取为新的列
    new_columns = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
                   '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']

    extracted_columns = exist_df['drink_labels'].apply(
        lambda x: [1 if item in x else 0 for item in new_columns]).tolist()
    extracted_df = pd.DataFrame(extracted_columns, columns=new_columns)
    ic(extracted_df.head())
    # 将提取的列添加到原始DataFrame中
    for c in new_columns:
        exist_df[c] = extracted_df[c]
    exist_df['labels_token'] = extracted_columns

    exist_df.to_csv(target_path, index=False, encoding='utf-8')
    # 统计新增列中值为1的数量
    column_counts = exist_df[new_columns].sum()
    print(column_counts)


if __name__ == '__main__':
    merge_multiple_categories('di_sku_log_data.csv', 'di_sku_log_data_temp.csv', [])
    process_data('di_sku_log_data_temp.csv', 'di_sku_log_drink_labels.csv', ['name', 'store_category', 'drink_labels'])