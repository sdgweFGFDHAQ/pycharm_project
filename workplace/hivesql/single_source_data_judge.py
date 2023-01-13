from pyhive import hive
from impala.dbapi import connect
from impala.util import as_pandas
import pandas as pd
import numpy as np
import time
import datetime
from dateutil.relativedelta import relativedelta


# 2对df判断半年内数据。根据数据分布设置（陈列活动、进货交易、拜访记录）次数阈值critical_weight
def get_threshold(x_df: pd.DataFrame, time_slot: int):
    c_6_month_df1 = x_df[x_df.create_time.apply(int) > time_slot]
    group_count1 = c_6_month_df1['storeid'].value_counts()
    data_frame1 = pd.DataFrame({'storeid': group_count1.index, 'count': group_count1}).reset_index()
    print(data_frame1)
    critical_weight = data_frame1['count'].median()
    print("次数阈值（中位数）:", critical_weight)
    return critical_weight


# 数据为时间戳字符串
def get_weight(x_df: pd.DataFrame, today_date):
    x_df['month'] = x_df['createtime'].apply(
        lambda x: (today_date - datetime.date.fromtimestamp(int(x) // 1000)).days / 30)
    x_df['weight'] = x_df['month'].apply(lambda x: 1 / 1 + 0.8 * x if x > 6 else 1.0)
    print(x_df)
    sum_weight = x_df[['storeid', 'weight']].groupby(by='storeid').sum()
    return sum_weight


# 数据为8位时间字符串
def get_weight_8(x_df: pd.DataFrame, today_date):
    x_df['createtime'] = pd.to_datetime(x_df['createtime'], format='%Y%m%d')
    x_df['month'] = x_df['createtime'].apply(
        lambda x: (today_date - x).days / 30)
    x_df['weight'] = x_df['month'].apply(lambda x: 1 / 1 + 0.8 * x if x > 6 else 1.0)
    print(x_df)
    sum_weight = x_df[['storeid', 'weight']].groupby(by='storeid').sum()
    return sum_weight


def judge_data(x_df1: pd.DataFrame, x_df2: pd.DataFrame, x_df3: pd.DataFrame):
    today_date = datetime.date.today()
    font_6_time = int(time.mktime(time.strptime(str(today_date - relativedelta(months=6)), "%Y-%m-%d")))
    critical_weight1 = get_threshold(x_df1, font_6_time)
    critical_weight2 = get_threshold(x_df2, font_6_time)
    critical_weight3 = get_threshold(x_df3, font_6_time)
    critical_weight = (critical_weight1 + critical_weight2 + critical_weight3) / 3
    # 3根据时间-权重映射函数计算单源企业数据的权重值weight
    sum_weight1 = get_weight(x_df1, today_date)
    sum_weight2 = get_weight(x_df2, today_date)
    sum_weight3 = get_weight_8(x_df3, today_date)
    # 对陈列、交易和拜访表赋权重
    sum_weight1['weight'] = sum_weight1['weight'].apply(lambda x: 0.2 * x)
    sum_weight2['weight'] = sum_weight2['weight'].apply(lambda x: 0.5 * x)
    sum_weight3['weight'] = sum_weight3['weight'].apply(lambda x: 0.3 * x)
    s_w = sum_weight1.add(sum_weight2, fill_value=0)
    sum_weight = s_w.add(sum_weight3, fill_value=0)
    # 4根据最终的sum_weight和critical_weight打标，字段为is_existence（是否存在），计算percentage（存在概率）
    percent_df = pd.DataFrame(sum_weight).reset_index()
    diff_num = percent_df['weight'].max() - percent_df['weight'].min()
    critical_percentage = (critical_weight - percent_df['weight'].min()) / diff_num
    percent_df['percentage'] = (percent_df['weight'] - percent_df['weight'].min()) / diff_num
    result_df = percent_df[percent_df.percentage >= critical_percentage]
    print(result_df)
    result_df.to_csv('result_df.csv')


if __name__ == '__main__':
    display_df = pd.read_csv('hn_display_project.csv', usecols=['storeid', 'createtime'])
    order_df = pd.read_csv('hn_receiving_note_v2.csv', usecols=['storeid', 'createtime'])
    visit_df = pd.read_csv('hn_receiving_note_v2.csv', usecols=['storeid', 'createtime'])
    judge_data(display_df, order_df, visit_df)

# from pyhive import hive
# conn = hive.Connection(host='192.168.0.150',port=10015,
# username='hive',password='xwbigdata2022',
# database='standard_db',auth='CUSTOM')
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM standard_db.di_store_dedupe LIMIT 10")
# print(cursor.fetchall())
# cursor.close()
# conn.close()
# a=0.2 b=0.5 c=0.3
