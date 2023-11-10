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


if __name__ == '__main__':
    # xlsx_to_csv_pd('洪山区0419.xlsx', '洪山区0419.csv')
    # csv = pd.read_csv('di_store_dl_hn_yy_costinfo_dedupe_202304191636.csv',
    #                   usecols=['storecode', 'storename', 'channel_type', 'channel', 'storelevel', 'region',
    #                            'is_point', 'cost_count', 'display_payamount', 'display_cost', 'purchase_pay_count',
    #                            'purchase_pay_number', 'purchase_pay_cash', 'registersnumber', 'area', 'cname',
    #                            'data_source'])
    report = ProfileReport(pd.read_csv('../../workplace/fewsamples/data/di_sku_log_drink_labels.csv'))
    report.to_file('di_sku_log_drink_labels.html')

    # xls_to_csv_pd('store_visit_report.xls', 'store_visit_report.csv')
    # columns = ['store_id', 'category1_new', 'category2_new', 'category3_new', 'name', 'predict_category', 'drink_label']
    # xlsx_to_csv_pd('sku_labels_fromlog.xlsx', 'sku_labels_fromlog.csv', columns)
    # set_file_standard_data('sku_labels_fromlog.csv', columns)

    # data = pd.read_csv('../../workplace/fewsamples/data/di_sku_log_drink_labels.csv', dtype={'store_id': str})
    # data.to_excel('C:\\Users\\86158\\Desktop\\di_sku_log_drink_labels2.xlsx')
