import pandas as pd
import time
import datetime
from dateutil.relativedelta import relativedelta
import math


# 2对df判断半年内数据。根据数据分布设置（陈列活动、进货交易、拜访记录）次数阈值critical_weight
def get_threshold(x_df: pd.DataFrame, time_slot: int):
    c_6_month_df1 = x_df[x_df.createtime.apply(int) > time_slot]
    group_count1 = c_6_month_df1['id'].value_counts()
    data_frame1 = pd.DataFrame({'id': group_count1.index, 'count': group_count1}).reset_index()
    print(data_frame1)
    critical_weight = data_frame1['count'].median()
    print("次数阈值（中位数）:", critical_weight)
    return critical_weight


# 数据为8位时间字符串
def get_threshold_8(x_df: pd.DataFrame, time_slot):
    ts = str(time_slot).replace('-', '')
    c_6_month_df1 = x_df[x_df.createtime.apply(str) > ts]
    group_count1 = c_6_month_df1['id'].value_counts()
    data_frame1 = pd.DataFrame({'id': group_count1.index, 'count': group_count1}).reset_index()
    print(data_frame1)
    critical_weight = data_frame1['count'].median()
    print("次数阈值（中位数）:", critical_weight)
    return critical_weight


# 数据为时间戳字符串
def get_weight(x_df: pd.DataFrame, today_date):
    x_df['month'] = x_df['createtime'].apply(
        lambda x: (today_date - datetime.date.fromtimestamp(int(x) // 1000)).days / 30)
    x_df['weight'] = x_df['month'].apply(lambda x: 0.6 ** (x - 6) if x > 6 else 1.0)
    print(x_df)
    sum_weight = x_df[['id', 'name', 'namepath', 'township', 'address', 'weight']].groupby(
        by=['id', 'name', 'namepath', 'township', 'address']).sum().reset_index()
    return sum_weight


# 数据为8位时间字符串
def get_weight_8(x_df: pd.DataFrame, today_date):
    x_df['createtime'] = pd.to_datetime(x_df['createtime'], format='%Y%m%d', errors='coerce')
    x_df['month'] = x_df['createtime'].apply(
        lambda x: (today_date - x).days / 30)
    x_df['weight'] = x_df['month'].apply(lambda x: 0.9 ** (x - 6) if x > 6 else 1.0)
    print(x_df)
    sum_weight = x_df[['id', 'name', 'namepath', 'township', 'address', 'weight']].groupby(
        by=['id', 'name', 'namepath', 'township', 'address']).sum().reset_index()
    return sum_weight


def judge_data(x_df1: pd.DataFrame, x_df2: pd.DataFrame, x_df3: pd.DataFrame):
    today_date = datetime.date.today()
    today_datetime = datetime.datetime.now()
    font_6_time = int(time.mktime(time.strptime(str(today_date - relativedelta(months=6)), "%Y-%m-%d")))
    critical_weight1 = get_threshold(x_df1, font_6_time)
    critical_weight2 = get_threshold(x_df2, font_6_time)
    critical_weight3 = get_threshold_8(x_df3, today_date - relativedelta(months=6))
    critical_weight = 0.2 * critical_weight1 + 0.5 * critical_weight2 + 0.3 * critical_weight3
    print('critical_weight', critical_weight1, ',', critical_weight2, ',', critical_weight3, ',', critical_weight)
    # 3根据时间-权重映射函数计算单源企业数据的权重值weight
    sum_weight1 = get_weight(x_df1, today_date)
    sum_weight2 = get_weight(x_df2, today_date)
    sum_weight3 = get_weight_8(x_df3, today_datetime)
    # 对陈列、交易和拜访表赋权重 a=0.2 b=0.5 c=0.3
    sum_weight1['weight'] = sum_weight1['weight'].apply(lambda x: 0.2 * x)
    sum_weight2['weight'] = sum_weight2['weight'].apply(lambda x: 0.5 * x)
    sum_weight3['weight'] = sum_weight3['weight'].apply(lambda x: 0.3 * x)
    # 合并数据框
    merged_df = pd.concat([sum_weight1, sum_weight2, sum_weight3], ignore_index=True)
    # 对 'weight' 列进行求和
    sum_weight = merged_df.groupby(['id', 'name', 'namepath', 'township', 'address'])['weight'].sum().reset_index()
    # 4根据最终的sum_weight和critical_weight打标，字段为is_existence（是否存在），计算percentage（存在概率）
    percent_df = sum_weight.copy()
    # ===设置阈值，降低最大值========
    # percent_df_max = percent_df[percent_df.weight > critical_weight]
    # print(percent_df_max.index.size)
    # percent_df_min = percent_df[percent_df.weight <= critical_weight]
    # print(percent_df_min.index.size)
    # df1 = separate_percent(percent_df_max, 0.9, 1)
    # df2 = separate_percent1(percent_df_min)
    # df_1_2 = pd.concat([df1, df2])
    df_1_2 = separate_percent1(percent_df)
    critical_percentage = math.atan(critical_weight) * 2 / math.pi
    print('置信率阈值critical_percentage:', critical_percentage)
    # result_df = df_1_2_3[df_1_2_3.percentage >= critical_percentage]
    result_df = df_1_2.sort_values(by='percentage', ascending=False).reset_index()
    print(result_df)
    result_df.to_csv('result_df.csv', columns=['id', 'name', 'namepath', 'township', 'address', 'percentage'],
                     index=False)


# Y=a+k(X-Min)
def separate_percent(df, a, b):
    diff_num = df['weight'].max() - df['weight'].min()
    print(df[df['weight'] == df['weight'].max()])
    print("最大值:", df['weight'].max(), "最小值:", df['weight'].min())
    df['percentage'] = df['weight'].apply(lambda x: a + (b - a) * (x - df['weight'].min()) / diff_num)
    # df['percentage'] = a + (b - a) * (df['weight'] - df['weight'].min()) / diff_num
    return df


# atan
def separate_percent1(df):
    df['percentage'] = df['weight'].apply(lambda x: math.atan(x) * 2 / math.pi)
    return df


if __name__ == '__main__':
    columns = ['id', 'name', 'namepath', 'township', 'address', 'createtime']
    display_df = pd.read_csv('display_data.csv')
    order_df = pd.read_csv('order_data.csv')
    visit_df = pd.read_csv('visit_data.csv', dtype={"createtime": str})
    print('--开始计算置信率--')
    judge_data(display_df, order_df, visit_df)
    print('--置信率计算完成--')
