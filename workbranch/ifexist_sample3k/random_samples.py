import os
import pandas as pd
from multiprocessing import Pool
import warnings
import os


# 用于重新切分店名，生成标准文件
def datafram_samples():
    cities = ['广州市', '深圳市', '武汉市', '上海城区', '杭州市']
    districts = ['海珠区', '南山区', '武昌区', '浦东新区', '钱塘区']
    drink = ['休闲餐饮', '早餐店', '烧烤店', '火锅店', '中式快餐店', '便利店', '大超市', '传统小店', '小超市',
             '西式快餐店', '自助餐', '批零',
             '大卖场', '餐厅', '自动售卖机']
    no_drink = ['母婴', '社区团购', '休闲娱乐场所', '医院', '化妆品集合连锁', '家居生活馆', '酒店宾馆', ]
    result = pd.DataFrame(
        columns=['id', 'name', 'namepath', 'city', 'district', 'township', 'address', 'geo_point', 'category1_new'])
    for i in range(len(cities)):
        path_sta = '/home/data/temp/zzx/workbranch/ifexist_sample3k/' + str(cities[i]) + '.csv'
        city_i = pd.read_csv(path_sta, index_col=0)
        csv_i = city_i[city_i['district'] == districts[i]]
        dr_df = csv_i[csv_i['category1_new'].isin(drink)]
        for key, group in dr_df.groupby('category1_new'):
            try:
                dd1 = group.sample(n=43, random_state=23)
            except Exception:
                dd1 = group
            print(cities[i], key, len(dd1.index))
            result = pd.concat([result, dd1], ignore_index=True)
        ndr_df = csv_i[csv_i['category1_new'].isin(no_drink)]
        for key, group in ndr_df.groupby('category1_new'):
            try:
                dd2 = group.sample(n=30)
            except Exception:
                dd2 = group
            print(cities[i], key, len(dd2.index))
            result = pd.concat([result, dd2], ignore_index=True)
    path_recall = '/home/data/temp/zzx/workbranch/ifexist_sample3k/ifexists_samples.csv'
    csv_recall = pd.read_csv(path_recall, index_col=0)
    csv_recall = csv_recall.groupby('district').sample(n=60, random_state=23)
    result_data = pd.concat([result, csv_recall], ignore_index=True)
    result_data.to_csv('/home/data/temp/zzx/workbranch/ifexist_sample3k/3000samples_v0.csv',
                       columns=['id', 'name', 'namepath', 'city', 'district', 'township', 'address', 'geo_point',
                                'category1_new'],
                       index=False)
    # 分区排序
    result_data = result_data.groupby(['district', 'township']).sample(frac=1)
    result_data.to_csv('/home/data/temp/zzx/workbranch/ifexist_sample3k/3000samples_v1.csv',
                       columns=['id', 'name', 'namepath', 'city', 'district', 'township', 'address', 'geo_point',
                                'category1_new'],
                       index=False)
    print(len(result_data.index))


def single_district():
    cities = ['杭州市']
    districts = ['钱塘区']
    drink = ['休闲餐饮', '早餐店', '烧烤店', '火锅店', '中式快餐店', '便利店', '大超市', '传统小店', '小超市',
             '西式快餐店', '自助餐', '批零',
             '大卖场', '餐厅', '自动售卖机']
    no_drink = ['母婴', '社区团购', '休闲娱乐场所', '医院', '化妆品集合连锁', '家居生活馆', '酒店宾馆']
    result = pd.DataFrame(
        columns=['id', 'name', 'namepath', 'city', 'district', 'township', 'address', 'geo_point', 'category1_new'])
    for i in range(len(cities)):
        path_sta = '/home/data/temp/zzx/workbranch/ifexist_sample3k/' + str(cities[i]) + '.csv'
        city_i = pd.read_csv(path_sta, index_col=0)
        csv_i = city_i[city_i['district'] == districts[i]]
        dr_df = csv_i[csv_i['category1_new'].isin(drink)]
        for key, group in dr_df.groupby('category1_new'):
            try:
                dd1 = group.sample(n=55, random_state=23)
            except Exception:
                dd1 = group
            print(cities[i], key, len(dd1.index))
            result = pd.concat([result, dd1], ignore_index=True)
        ndr_df = csv_i[csv_i['category1_new'].isin(no_drink)]
        for key, group in ndr_df.groupby('category1_new'):
            try:
                dd2 = group.sample(n=30)
            except Exception:
                dd2 = group
            print(cities[i], key, len(dd2.index))
            result = pd.concat([result, dd2], ignore_index=True)
    path_recall = '/home/data/temp/zzx/workbranch/ifexist_sample3k/ie_samples.csv'
    csv_recall = pd.read_csv(path_recall, index_col=0)
    csv_recall = csv_recall.groupby('district').sample(n=70, random_state=23)
    result_data = pd.concat([result, csv_recall], ignore_index=True)
    result_data.to_csv('/home/data/temp/zzx/workbranch/ifexist_sample3k/samples_v1.0.csv',
                       columns=['id', 'name', 'namepath', 'city', 'district', 'township', 'address', 'geo_point',
                                'category1_new'],
                       index=False)
    # 分区排序
    result_data = result_data.groupby(['township']).sample(frac=1)
    result_data.to_csv('/home/data/temp/zzx/workbranch/ifexist_sample3k/samples_v1.1.csv',
                       columns=['id', 'name', 'namepath', 'city', 'district', 'township', 'address', 'geo_point',
                                'category1_new'],
                       index=False)
    print('总数据量：', len(result_data))


if __name__ == '__main__':
    # datafram_samples()
    single_district()
