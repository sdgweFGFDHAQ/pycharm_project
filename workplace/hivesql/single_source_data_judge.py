from pyhive import hive
from impala.dbapi import connect
from impala.util import as_pandas
import pandas as pd
import numpy as np
import time
import datetime
from dateutil.relativedelta import relativedelta


def get_connection(c_time, x_table):
    conn = connect(host='124.71.220.115',  # 主机
                   port=10015,  # 端口
                   auth_mechanism='PLAIN',
                   user='hive',  # 用户
                   password='xwbigdata2022',
                   database='standard_db'  # 数据库
                   )
    # 1获取目标表中，在di_store_dedupe表中存在且单源的数据df
    cursor = conn.cursor()
    sql = "select storeid, {0} from standard_db.{1} " \
          "where storeid in (select original_id from standard_db.di_store_dedupe where appcode not like '%,%') " \
          "limit 10".format(c_time, x_table)
    cursor.execute(sql)
    # cursor.fetchall()
    yy_df = as_pandas(cursor)
    # 关闭连接
    cursor.close()
    conn.close()
    yy_df.to_csv('yy_data.csv')
    return yy_df


def judge_data(x_df: pd.DataFrame):
    # 2对df判断半年内数据。根据数据分布设置（陈列活动、进货交易）次数阈值critical_weight
    today_date = datetime.date.today()
    font_6_time = int(time.mktime(time.strptime(str(today_date - relativedelta(months=6)), "%Y-%m-%d")))
    c_6_month_df = x_df[x_df.createtime.apply(int) > font_6_time]
    group_count = c_6_month_df['storeid'].value_counts()
    data_frame = pd.DataFrame({'storeid': group_count.index, 'count': group_count}).reset_index()
    print(data_frame)
    critical_weight = data_frame['count'].median()
    print("次数阈值（中位数）:", critical_weight)
    # 3根据时间-权重映射函数计算单源企业数据的权重值weight
    x_df['month'] = x_df['createtime'].apply(
        lambda x: (today_date - datetime.date.fromtimestamp(int(x) // 1000)).days / 30)
    x_df['weight'] = x_df['month'].apply(lambda x: 1 / 1 + 0.8 * x if x > 6 else 1.0)
    print(x_df)
    sum_weight = x_df[['storeid', 'weight']].groupby(by='storeid').sum()
    # 4根据最终的sum_weight和critical_weight打标，字段为is_existence（是否存在），计算percentage（存在概率）
    percent_df = pd.DataFrame(sum_weight).reset_index()
    diff_num = percent_df['weight'].max() - percent_df['weight'].min()
    critical_percentage = (critical_weight - percent_df['weight'].min()) / diff_num
    percent_df['percentage'] = (percent_df['weight'] - percent_df['weight'].min()) / diff_num
    result_df = percent_df[percent_df.percentage >= critical_percentage]
    print(result_df)
    result_df.to_csv('result_df.csv')


def get_connection_s(ti_list, ta_list):
    conn = connect(host='124.71.220.115',  # 主机
                   port=10015,  # 端口
                   auth_mechanism='PLAIN',
                   user='hive',  # 用户
                   password='xwbigdata2022',
                   database='standard_db'  # 数据库
                   )
    # 1获取目标表中，在di_store_dedupe表中存在且单源的数据df
    cursor = conn.cursor()
    sql = "select storeid, {0} from standard_db.{1} " \
          "where storeid in (select original_id from standard_db.di_store_dedupe where appcode not like '%,%') " \
          "limit 100000".format(ti_list[1], ta_list[0])
    cursor.execute(sql)
    hn_df1 = as_pandas(cursor)
    hn_df1.to_csv('hn_data1.csv')
    sql = "select storeid, {0} from standard_db.{1} " \
          "where storeid in (select original_id from standard_db.di_store_dedupe where appcode not like '%,%') " \
          "limit 100000".format(ti_list[0], ta_list[1])
    cursor.execute(sql)
    hn_df2 = as_pandas(cursor)
    hn_df2.to_csv('hn_data2.csv')
    # 关闭连接
    cursor.close()
    conn.close()
    return hn_df1, hn_df2


def judge_data_s(x_df1: pd.DataFrame, x_df2: pd.DataFrame):
    # 2对df判断半年内数据。根据数据分布设置（陈列活动、进货交易）次数阈值critical_weight
    today_date = datetime.date.today()
    font_6_time = int(time.mktime(time.strptime(str(today_date - relativedelta(months=6)), "%Y-%m-%d")))
    # 表1
    c_6_month_df1 = x_df1[x_df1.create_time.apply(int) > font_6_time]
    group_count1 = c_6_month_df1['storeid'].value_counts()
    data_frame1 = pd.DataFrame({'storeid': group_count1.index, 'count': group_count1}).reset_index()
    print(data_frame1)
    critical_weight1 = data_frame1['count'].median()
    print("次数阈值（中位数）:", critical_weight1)
    # 3根据时间-权重映射函数计算单源企业数据的权重值weight
    x_df1['month'] = x_df1['create_time'].apply(
        lambda x: (today_date - datetime.date.fromtimestamp(int(x) // 1000)).days / 30)
    x_df1['weight'] = x_df1['month'].apply(lambda x: 1 / 1 + 0.8 * x if x > 6 else 1.0)
    print(x_df1)
    sum_weight1 = x_df1[['storeid', 'weight']].groupby(by='storeid').sum()
    # 表2
    c_6_month_df2 = x_df2[x_df2.createtime.apply(int) > font_6_time]
    group_count2 = c_6_month_df2['storeid'].value_counts()
    data_frame2 = pd.DataFrame({'storeid': group_count2.index, 'count': group_count2}).reset_index()
    print(data_frame2)
    critical_weight2 = data_frame2['count'].median()
    print("次数阈值（中位数）:", critical_weight2)
    # 3根据时间-权重映射函数计算单源企业数据的权重值weight
    x_df2['month'] = x_df2['createtime'].apply(
        lambda x: (today_date - datetime.date.fromtimestamp(int(x) // 1000)).days / 30)
    x_df2['weight'] = x_df2['month'].apply(lambda x: 1 / 1 + 0.8 * x if x > 6 else 1.0)
    print(x_df2)
    sum_weight2 = x_df2[['storeid', 'weight']].groupby(by='storeid').sum()
    # 对陈列和交易表赋权重
    sum_weight1['weight'] = sum_weight1['weight'].apply(lambda x: 0.3 * x)
    sum_weight2['weight'] = sum_weight2['weight'].apply(lambda x: 0.7 * x)
    sum_weight = sum_weight1.add(sum_weight2, fill_value=0)
    critical_weight = (critical_weight1 + critical_weight2)/2
    # 4根据最终的sum_weight和critical_weight打标，字段为is_existence（是否存在），计算percentage（存在概率）
    percent_df = pd.DataFrame(sum_weight).reset_index()
    diff_num = percent_df['weight'].max() - percent_df['weight'].min()
    critical_percentage = (critical_weight - percent_df['weight'].min()) / diff_num
    percent_df['percentage'] = (percent_df['weight'] - percent_df['weight'].min()) / diff_num
    result_df = percent_df[percent_df.percentage >= critical_percentage]
    print(result_df)
    result_df.to_csv('result_df_hn.csv')


if __name__ == '__main__':
    # c_time = 'createtime'
    # yy_table = 'di_store_yy_display'
    # # yy_df = get_connection(c_time, yy_table)
    # yy_df = pd.read_csv('yy_data.csv', usecols=['storeid', 'createtime'])
    # judge_data(yy_df)

    time_list = ['createtime', 'create_time']
    table_list = ['di_store_hn_display_project', 'di_store_hn_display_purchase']
    hn_df1, hn_df2 = get_connection_s(time_list, table_list)
    # hn_df1 = pd.read_csv('hn_data1.csv', usecols=['storeid', 'createtime'])
    # hn_df2 = pd.read_csv('hn_data2.csv', usecols=['storeid', 'createtime'])
    # judge_data_s(hn_df1, hn_df2)

    # from pyhive import hive
    # conn = hive.Connection(host='192.168.0.150',port=10015,
    # username='hive',password='xwbigdata2022',
    # database='standard_db',auth='CUSTOM')
    # cursor = conn.cursor()
    # cursor.execute("SELECT * FROM standard_db.di_store_dedupe LIMIT 10")
    # print(cursor.fetchall())
    # cursor.close()
    # conn.close()
