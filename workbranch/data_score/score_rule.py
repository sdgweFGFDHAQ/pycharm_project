import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport


# 处理数据文件格式
def xlsx_to_csv_pd(source, target):
    data_xls = pd.read_excel(source, index_col=0)
    data_xls.to_csv(target, encoding='utf-8')

    # csvv = pd.read_csv(target,
    #                    usecols=['id', 'name', 'address', 'category1_new', 'category2_new', 'visit_num_6m', 'cost',
    #                             'area', 'is_point', 'is_direct', 'is_freezer'])
    # csvv.to_csv(target,
    #             columns=['id', 'name', 'address', 'category1_new', 'category2_new', 'visit_num_6m', 'cost',
    #                      'area', 'is_point', 'is_direct', 'is_freezer'])
    csvv = pd.read_csv(target)
    csvv.to_csv(target)

    csv2 = pd.read_csv(target)
    print(csv2)


# 处理数据文件格式
def xls_to_csv_pd(source, target):
    data_xls = pd.read_excel(source, index_col=0)
    data_xls.to_csv(target, encoding='utf-8')

    # csvv = pd.read_csv(target)
    # csvv.to_csv(target)
    #
    # csv2 = pd.read_csv(target)
    # print(csv2)


if __name__ == '__main__':
    # xlsx_to_csv_pd('洪山区0419.xlsx', '洪山区0419.csv')
    # csv = pd.read_csv('di_store_dl_hn_yy_costinfo_dedupe_202304191636.csv',
    #                   usecols=['storecode', 'storename', 'channel_type', 'channel', 'storelevel', 'region',
    #                            'is_point', 'cost_count', 'display_payamount', 'display_cost', 'purchase_pay_count',
    #                            'purchase_pay_number', 'purchase_pay_cash', 'registersnumber', 'area', 'cname',
    #                            'data_source'])
    # report = ProfileReport(csv)
    # report.to_file('洪山区0419.html')

    xls_to_csv_pd('store_visit_report.xls', 'store_visit_report.csv')
