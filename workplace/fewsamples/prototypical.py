import random

import pandas as pd
from icecream.icecream import ic
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from models.proto_model import ProtoTypicalNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrian_bert_url = "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese"
labeled_path = '../sv_report_data.csv'
unlabeled_path = '../unlabeled_data.csv'

token_max_length = 12
batch_size = 64
epochs = 30


def get_Support_Query(train_df, labels, k=10):
    def minimize_set(df_i, count_list, excessive_list):
        tally = count_list.copy()
        tally = [i - j for i, j in zip(tally, df_i['multi_label'])]
        if k - 1 not in count_list:
            excessive_list.append(df_i['index_col'])
            count_list.clear()
            count_list.extend(tally)

    df = train_df.copy()
    df['multi_label'] = df[labels].values.tolist()
    df['index_col'] = df.index

    # 随机抽样
    support_df, add_list = pd.DataFrame(), []
    label_number = len(labels)
    count = [0] * label_number
    for label_i in range(label_number):
        # 取出当前标签的df
        label_i_df = df[df[labels[label_i]] == 1]
        # 满足每个标签最少出现K次，如果原始数据集df不足K条则结束
        while count[label_i] < k and label_i_df.shape[0] > 0:
            add_series = label_i_df.sample(n=1)
            label_i_df.drop(add_series.index, inplace=True)
            add_list.append(add_series)
            count = [i + j for i, j in zip(count, add_series['multi_label'].values[0])]
    support_df = pd.concat([support_df] + add_list)

    # 删除多余的数据
    delete_list = []
    support_df.apply(minimize_set, args=(count, delete_list), axis=1)
    support_df.drop(delete_list, inplace=True)
    return support_df


def get_Nway_Kshot(df, category_list, way, shot, query):
    # 随机选择3个分类
    selected_categories = random.sample(category_list, way)

    # 创建一个空的DataFrame用于存储支持集和查询集
    support_set = pd.DataFrame()
    query_set = pd.DataFrame()

    # 遍历选择的分类
    for category in selected_categories:
        # 获取该分类下的数据
        category_data = df[df[category] == 1]

        # 随机选择n个样本作为支持集
        support_samples = category_data.sample(n=shot)
        support_set = pd.concat([support_set, support_samples])

        # 从该分类中移除支持集样本，剩下的样本作为查询集
        category_data = category_data.drop(support_samples.index)

        # 随机选择n个样本作为查询集
        query_samples = category_data.sample(n=query)
        query_set = pd.concat([query_set, query_samples])

    print("Support Set and Query Set segmentation completed")
    return support_set, query_set


def get_dataset(df, category_list, way):
    # 创建一个空的DataFrame用于存储支持集和查询集
    data_set = pd.DataFrame()

    # 遍历选择的分类
    for category in category_list:
        # 获取该分类下的数据
        category_data = df[df[category] == 1]

        # 随机选择n个样本作为支持集
        samples = category_data.sample(n=way)
        data_set = pd.concat([data_set, samples])

    train_set, test_set = train_test_split(data_set, random_state=0.2)
    print("Train Set and Test Set segmentation completed")
    return train_set, test_set


# 线下跑店(店铺存在且售品) 1399条
def get_labeled_dataloader(df, bert_tokenizer, label_list):
    # 生成类别-id字典
    # df['cat_id'] = df['storeType'].factorize()[0]
    # cat_df = df[['storeType', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
    # cat_df.to_csv('./data/store_type_to_id.csv')

    # 创建输入数据的空列表
    input_ids = []
    attention_masks = []
    label2id_list = []
    # 遍历数据集的每一行
    for index, row in df.iterrows():
        # 处理特征
        encoded_dict = bert_tokenizer.encode_plus(
            row['name'],
            row['storeType'],
            add_special_tokens=True,
            max_length=16,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'].squeeze())
        attention_masks.append(encoded_dict['attention_mask'].squeeze())

        # 处理类别
        labels_tensor = torch.tensor([row[label] for label in label_list])
        label2id_list.append(labels_tensor)

    dataset = TensorDataset(torch.stack(input_ids), torch.stack(attention_masks), torch.stack(label2id_list))
    dataloader = DataLoader(dataset, batch_size=df.shape[0], shuffle=False)
    return dataloader


# 泛样本127w
def get_unlabeled_dataloader(file_path, bert_tokenizer):
    unlabeled_df = pd.read_csv(file_path, usecols=['store_id', 'name', 'category3_new'])
    # cat2id_df = pd.read_csv('./data/category_to_id.csv')
    # cat2id = dict(zip(cat2id_df['category3_new'], cat2id_df['cat_id']))

    # 创建输入数据的空列表
    input_ids = []
    attention_masks = []

    # 遍历数据集的每一行
    for index, row in unlabeled_df.iterrows():
        # 处理特征
        encoded_dict = bert_tokenizer.encode_plus(
            row['name'],
            row['storeType'],
            add_special_tokens=True,
            max_length=16,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'].squeeze())
        attention_masks.append(encoded_dict['attention_mask'].squeeze())

    # 创建TensorDataset对象
    dataset = TensorDataset(torch.stack(input_ids), torch.stack(attention_masks))
    # 创建DataLoader对象
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def accuracy(pred_y, y):
    predicted_labels = (pred_y > 0.5).float()  # 使用0.5作为阈值，大于阈值的为预测为正类
    acc = (predicted_labels == y).float().mean()
    return acc


def multilabel_categorical_crossentropy(y_pred, y_true):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12

    zeros = torch.zeros_like(y_pred[..., :1])

    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return torch.sum(neg_loss + pos_loss)


def training(support_loader, query_loader, model):
    criterion = nn.BCEWithLogitsLoss()
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
    model.train()
    epoch_los, epoch_acc = 0.0, 0.0
    for i, (support_input, query_input) in enumerate(zip(support_loader, query_loader)):
        # 1. 放到GPU上
        support_input0 = support_input[0].to(device, dtype=torch.long)
        query_input0 = query_input[0].to(device, dtype=torch.long)
        query_input2 = query_input[2].to(device, dtype=torch.long)
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        output = model(support_input0, query_input0)
        # outputs = outputs.squeeze(1)
        # 4. 计算损失
        loss = multilabel_categorical_crossentropy(output, query_input2.float())
        # loss = torch.sum(output, dim=0)
        epoch_los += loss.item()
        # 5.预测结果
        accu = accuracy(output, query_input2)
        epoch_acc += accu.item()
        # 6. 反向传播
        loss.requires_grad_(True)
        loss.backward()
        # 7. 更新梯度
        optimizer.step()
    loss_value = epoch_los / len(support_loader)
    acc_value = epoch_acc / len(support_loader)
    print("accuracy: {:.2%},loss:{:.4f}".format(acc_value, loss_value))
    return acc_value, loss_value


def evaluating(support_loader, query_loader, model):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    with torch.no_grad():
        epoch_los, epoch_acc = 0.0, 0.0
        for i, (support_input, query_input) in enumerate(zip(support_loader, query_loader)):
            # 1. 放到GPU上
            support_input0 = support_input[0].to(device, dtype=torch.long)
            query_input0 = query_input[0].to(device, dtype=torch.long)
            query_input2 = query_input[2].to(device, dtype=torch.long)
            # 2. 计算输出
            output = model(support_input0, query_input0)
            # outputs = outputs.squeeze(1)
            # 3. 计算损失
            loss = multilabel_categorical_crossentropy(output, query_input2.float())
            # loss = torch.sum(output, dim=0)
            epoch_los += loss.item()
            # 4.预测结果
            accu = accuracy(output, query_input2)
            epoch_acc += accu.item()
        loss_value = epoch_los / len(support_loader)
        acc_value = epoch_acc / len(support_loader)
        print("accuracy: {:.2%},loss:{:.4f}".format(acc_value, loss_value))
    return acc_value, loss_value


if __name__ == '__main__':
    features = ['name', 'storeType']
    labels = ['碳酸饮料', '果汁', '茶饮', '水', '乳制品', '植物蛋白饮料', '功能饮料']
    columns = ['store_id', 'drinkTypes']
    columns.extend(features)
    columns.extend(labels)

    labeled_df = pd.read_csv(labeled_path, usecols=columns)
    # # bert_config = AutoConfig.from_pretrained(pretrian_bert_url + '/config.json')

    # tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)
    # bert_layer = AutoModel.from_pretrained(pretrian_bert_url)
    # proto_model = ProtoTypicalNet(
    #     bert_layer=bert_layer,
    #     input_dim=768,
    #     hidden_dim=768,
    #     num_class=len(labels)
    # ).to(device)
    #
    # # 采用NwayKshot采样
    # for i in range(3):
    #     support_df, query_df = get_Nway_Kshot(labeled_df, labels, 7, 32, 8)
    #     support_dataloader = get_labeled_dataloader(support_df, tokenizer, labels)
    #     query_dataloader = get_labeled_dataloader(query_df, tokenizer, labels)
    #     for j in range(epochs):
    #         training(support_dataloader, query_dataloader, proto_model)
    # 采用最小包含算法采样
    train_set, test_set = train_test_split(labeled_df, test_size=0.2)
    print('train_set len:{} test_set len:{}'.format(train_set.shape[0], test_set.shape[0]))
    support_set = get_Support_Query(train_set, labels, k=2000)
    query_set = train_set.drop(support_set.index)
    print('support_set len:{} query_set len:{}'.format(support_set.shape[0], query_set.shape[0]))
    # get_unlabeled_dataloader(unlabeled_path, tokenizer)
