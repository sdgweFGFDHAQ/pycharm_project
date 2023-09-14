import numpy as np
import pandas as pd


def predict_acc(df):
    # 过滤空
    print("统计不加入'其他类'的分类准确率：")
    df = df[(df['category1_new'].notna() & df['category1_new'].notnull())]
    number = df.shape[0]
    # 分组数据并计算准确度
    grouped = df.groupby('category1_new')

    name_lists = []
    accuracy_list = []
    for group_name, group_data in grouped:
        # 统计匹配数量
        match_count = (group_data['category1_new'] == group_data['predict_category']).sum()
        # 统计总记录数量
        total_count = len(group_data)
        # 计算准确度
        accuracy = match_count / total_count

        name_lists.append(group_name)
        accuracy_list.append(accuracy)
        # print('原始分类准确率{}: {}'.format(group_name, accuracy))

    # 汇总准确度
    overall_accuracy = sum(accuracy_list) / len(accuracy_list)
    print(f"Overall Accuracy: {overall_accuracy}")
    name_lists.append('OverallAccuracy')
    accuracy_list.append(overall_accuracy)
    df = pd.DataFrame({'category': name_lists, 'accuracy': accuracy_list})

    return df, number


def threshold_acc(df):
    # 过滤空
    print("统计根据阈值设置'其他类'的分类准确率：")
    df = df[(df['category1_new'].notna() & df['category1_new'].notnull())]
    # 过滤其他
    df = df[(df['threshold_category'].notna() & df['threshold_category'].notnull())]
    number = df.shape[0]
    grouped = df.groupby('category1_new')

    name_lists = []
    accuracy_list = []
    for group_name, group_data in grouped:
        # 统计匹配数量
        match_count = (group_data['category1_new'] == group_data['threshold_category']).sum()
        # 统计总记录数量
        total_count = len(group_data)
        # 计算准确度
        accuracy = match_count / total_count

        name_lists.append(group_name)
        accuracy_list.append(accuracy)
        # print('阈值准确率{}:{}'.format(group_name, accuracy))

    # 汇总准确度
    overall_accuracy = sum(accuracy_list) / len(accuracy_list)
    print(f"Overall Accuracy: {overall_accuracy}")
    name_lists.append('OverallAccuracy')
    accuracy_list.append(overall_accuracy)
    df = pd.DataFrame({'category': name_lists, 'accuracy': accuracy_list})

    return df, number


def other(df):
    # 根据阈值的值，分区间统计'其他类'的分类效果
    print('')


def cal_CK_data(methed):
    # 创建一个空列表用于存储生成的 DataFrame
    name_values = {}
    all_number, all_cal_number = 0, 0
    for i in range(8):
        # 读取 CSV 文件
        df = pd.read_csv('C:\\Users\\86158\\Desktop\\fsdownload\\cal_CK_data\\predict_category_' + str(i) + '.csv')
        df_number = df.shape[0]
        all_number += df_number
        r_df, cal_number = methed(df)
        all_cal_number += cal_number

        df_dict = dict(zip(r_df['category'], r_df['accuracy']))
        # 遍历df_dict,如果key在name_values中，value相加，不在则添加到name_values
        for key, value in df_dict.items():
            if key in name_values:
                name_values[key] += value
            else:
                name_values[key] = value
    for key in name_values:
        name_values[key] /= 8

    result = pd.DataFrame(list(name_values.items()), columns=['category', 'accuracy'])
    print('all_number:{}, cal_number:{}'.format(all_number, all_cal_number))
    result.to_excel('result/category_' + str(methed.__name__) + '.xlsx', index=False)


if __name__ == '__main__':
    cal_CK_data(methed=predict_acc)
    cal_CK_data(methed=threshold_acc)
