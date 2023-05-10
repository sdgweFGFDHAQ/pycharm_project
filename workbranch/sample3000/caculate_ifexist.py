import numpy as np
import pandas as pd


def run1():
    csv_4s = pd.read_csv('3000samples_v0.csv')
    df1 = csv_4s.tail(n=300)
    df1 = df1[df1['district'] != '上城区']
    csv_hz = pd.read_csv('samples_v1.0.csv')
    df2 = csv_hz.tail(n=70)
    result = pd.concat((df1, df2))
    result.to_csv('test_ifexist.csv', index=False)


def run2():
    # id
    csv1 = pd.read_csv('test_ifexist.csv', usecols=['store_id'])
    # csv1['store_id'] = csv1['store_id'].astype('str')

    # id
    csv2 = pd.read_csv('new_sv_report.csv', usecols=['store_id', 'is_exist'])
    # csv2['store_id'] = csv2['store_id'].astype('str')

    # original_id
    csv3 = pd.read_csv('dedupe_ifexists_inner.csv', usecols=['store_id', 'original_id', 'percentage'])
    # csv3['store_id'] = csv3['store_id'].astype('str')
    # csv3.to_csv('dedupe_ifexists.csv')

    res = pd.merge(csv1, csv2, how='left', on='store_id')

    res = pd.merge(res, csv3, how='left', on='store_id')
    res.to_csv('tn_res.csv')


def run3():
    # 跑出结果为:total=310 t=123 f=80 none=107
    res = pd.read_csv('tn_res.csv', index_col=0)
    print(res.head(3))
    res = res[res['is_exist'].notna()]
    print(len(res))
    # 203-117=86
    res = res[res['percentage'].notna()]
    print('给200条左右,跑出结果为：', len(res))
    res.to_csv('tnr_result.csv')
    true_num = len(res[res['is_exist'] == 't'])
    print(true_num / len(res))
    false_num = len(res[res['is_exist'] == 'f'])
    print(false_num / len(res))


if __name__ == '__main__':
    run2()
