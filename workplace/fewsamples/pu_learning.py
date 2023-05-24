import numpy
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


"""
PU-Learning，即P代表的是Positive，U代表的是Unlabel，负样本实际上是泛样本
"""
noEDA_prefix = '/home/data/temp/lxb/alchemy/data/noEDA_data'
input_prefix = '/home/data/temp/lxb/alchemy/data/input_dataset'


# 生成数据对
def generate_data_pair(positive_df, negative_df):
    positive_df['sign'], negative_df['sign'] = 1, 1

    p_samples = positive_df.merge(positive_df, how='left', on='sign')
    p_samples.drop(columns='sign')
    p_samples['label'] = 1
    print(p_samples.head())
    n_samples = positive_df.merge(negative_df, how='left', on='sign')
    n_samples.drop(columns='sign')
    n_samples['label'] = 0

    result_df = pd.concat((p_samples, n_samples))
    result_df = result_df.rename(columns={0: 'name1', 1: 'category1', 2: "name2", 3: "category2", 4: "label"})
    return result_df


# (1)使用正样本和泛洋本训练分类器
def train_by_all_dataset(source_path):
    # 正样本数据集
    pos_df = pd.read_csv(source_path + '/Pos_df.csv', usecols=['name', 'storeType'])
    # 泛样本数据集
    unl_df = pd.read_csv(source_path + '/Neg_df.csv', usecols=['name', 'storeType'])

    # 欠采样，采用笛卡尔积构成样本对
    num = min(pos_df.shape[0], unl_df.shape[0])
    print("设置抽取正样本数据量num:{}".format(num))
    pos_df, neg_df = pos_df.sample(n=num), unl_df.sample(n=num * 2)

    init_dataset = generate_data_pair(pos_df, neg_df)
    init_dataset.to_csv(source_path + '/pu_train_df.csv', index=False)
    # 训练，保存模型
    model = run_train_model(source_path + '/pu_train_df.csv', is_init=True)
    torch.save(model, 'bert_attention.model')
    acc_dict = torch.load("test_acc_history.pth")
    print(acc_dict)


# (2)对泛样本打分，选取概率最高的作为负样本，重新生成样本集csv文件
def score(pos_path, pred_path):
    # 构建预测集
    pos_data = pd.read_csv(pos_path + '/Pos_df.csv', usecols=['name', 'storeType'])
    pred_path = pd.read_csv(pred_path + '/Neg_df.csv', usecols=['name', 'storeType'])
    positive_df['label'] = 1
    p_samples = positive_df.merge(positive_df, how='left', on='label')
    print(p_samples.head())
    # 模型预测
    model = torch.load('bert_attention.model')
    predict(model, pred_dataloader, pos_path=positive_path, pred_path=predict_path)

def rerun(epoch):
    acc_List = []
    init_model = True
    for _ in range(epoch):
        extract_negative_dataset(noEDA_prefix, input_prefix)
        get_train_and_test(input_prefix, input_prefix)
        acc = train_and_test(source_path=input_prefix, is_init=init_model)
        acc_List.append(acc)

        init_model = False
    # (4)最终我们拿到了N次的预测结果，取平均作为最终的预测概率
    mean_acc = numpy.mean(acc_List)
    print(mean_acc)


if __name__ == '__main__':
    rerun(5)
