import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport


# 处理数据文件格式
def xlsx_to_csv_pd(source, target, use_columns):
    data_xls = pd.read_excel(source, index_col=0)
    data_xls.to_csv(target, encoding='utf-8')

    csvv = pd.read_csv(target, usecols=use_columns)
    print(csvv.head())
    # csvv.columns = ['store_id', 'sku_code', 'brand_name', 'series_name', 'sku_name', 'category1_new', 'category2_new',
    #                 'category3_new', 'name', 'predict_category', 'drink_label']
    csvv.to_csv(target)


def set_file_standard_data(path, use_columns):
    csv_data = pd.read_csv(path, usecols=use_columns)
    csv_data.drop_duplicates(keep='first', inplace=True)
    print(csv_data.shape[0])

    csv_data['store_category'] = csv_data['category3_new']

    # 用一级标签填充空白(NAN)的二级标签、三级标签
    # 删除至少有3个NaN值的行 # data = data.dropna(axis=0, thresh=3)
    csv_data['category2_new'].fillna(csv_data['category1_new'], inplace=True)
    csv_data['store_category'].fillna(csv_data['category2_new'], inplace=True)
    csv_data['store_category'].fillna(csv_data['predict_category'], inplace=True)

    csv_data.to_csv('sku_fromlog_new_cate.csv', columns=use_columns + ['store_category'])


if __name__ == '__main__':
    # xlsx_to_csv_pd('洪山区0419.xlsx', '洪山区0419.csv')
    # csv = pd.read_csv('di_store_dl_hn_yy_costinfo_dedupe_202304191636.csv',
    #                   usecols=['storecode', 'storename', 'channel_type', 'channel', 'storelevel', 'region',
    #                            'is_point', 'cost_count', 'display_payamount', 'display_cost', 'purchase_pay_count',
    #                            'purchase_pay_number', 'purchase_pay_cash', 'registersnumber', 'area', 'cname',
    #                            'data_source'])
    # report = ProfileReport(csv)
    # report.to_file('洪山区0419.html')

    # xls_to_csv_pd('store_visit_report.xls', 'store_visit_report.csv')
    columns = ['store_id', 'category1_new', 'category2_new', 'category3_new', 'name', 'predict_category', 'drink_label']
    xlsx_to_csv_pd('sku_labels_fromlog.xlsx', 'sku_labels_fromlog.csv', columns)
    set_file_standard_data('sku_labels_fromlog.csv', columns)
