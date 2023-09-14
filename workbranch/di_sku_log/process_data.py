import pandas as pd


def merge_multiple_categories(csv_data, target_path, keep_cols, merge_col):
    print("======合并同store_id的sku商品=======")
    print("dedupe 原始sku数据集：", csv_data.shape[0])
    csv_data.drop_duplicates(keep='first', inplace=True)
    print("dedupe 原始sku去重：", csv_data.shape[0])
    # 合并原始类别和预测类别
    csv_data['storetype'] = csv_data['category1_new'].fillna(csv_data['predict_category'])

    save_cols = keep_cols + [merge_col]
    csv_result = csv_data[save_cols]
    csv_result = csv_result.set_index(keep_cols)

    # 合并售卖的商品
    grouped = csv_result.groupby(by=keep_cols)
    result = grouped.agg({merge_col: lambda x: list(set(item for item in x if item != ''))})
    result.reset_index(inplace=True)
    print("合并饮料商品：", result.shape[0])

    print("========提取0-1饮料标签=========")
    exist_df = result
    print("合并饮料sku数据集：", exist_df.shape[0])
    # 去除storetype为空的数据
    exist_df = exist_df.dropna(subset=['storetype'])
    exist_df = exist_df[exist_df['storetype'].notnull() & (exist_df['storetype'] != '')].reset_index(drop=True)
    print("合并饮料sku删除空值：", exist_df.shape[0])

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
    read_path = 'C:\\Users\\86158\\Desktop\\a_di_store_dedupe_labeling_202309071810.csv'

    save_path = 'C:\\Users\\86158\\Desktop\\di_sku_log_drink_labeling_zzx.csv'
    use_columns = ['id', 'name', 'appcode', 'category1_new', 'predict_category', 'drink_label']
    keep_columns = ['id', 'name', 'appcode', 'storetype']
    merge_column = 'drink_label'

    # 集成学习预测品类标签，对相应数据集进行处理
    dfs = []
    for i in range(6):
        read_path = "./data/di_sku_log_drink_data_{}.csv".format(i)
        df = pd.read_csv(read_path, usecols=use_columns, keep_default_na=False)
        print("======第{}个csv文件，数据量为{}=======".format(i, df.shape[0]))
        dfs.append(df)

    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    save_path = "./data/di_sku_log_drink_labeling_zzx.csv"
    merge_multiple_categories(merged_df, save_path, keep_columns, merge_column)
