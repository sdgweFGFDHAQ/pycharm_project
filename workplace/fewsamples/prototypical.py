import random

import numpy as np
import pandas as pd
from icecream.icecream import ic
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter

from models.proto_model import ProtoTypicalNet
from models.proto_model_2 import ProtoTypicalNet2
from workplace.fewsamples.preprocess_data import Preprocess
from workplace.fewsamples.utils.mini_tool import WordSegment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./logs/v1')

pretrian_bert_url = "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese"

labeled_path = '../sv_report_data.csv'
labeled_di_sku_path = './data/di_sku_log_drink_labels.csv'
unlabeled_path = '../unlabeled_data.csv'

token_max_length = 12
batch_size = 15
epochs = 50


def get_Support_Query(train_df, label_list, k=10):
    def minimize_set(df_i, count_list, excessive_list):
        tally = count_list.copy()
        tally = [i - j for i, j in zip(tally, df_i['multi_label'])]
        if (min(count_list) >= k) and (k - 1 not in count_list):
            excessive_list.append(df_i['index_col'])
            count_list.clear()
            count_list.extend(tally)

    df = train_df.copy()
    df['multi_label'] = df[label_list].values.tolist()
    df['index_col'] = df.index

    # 随机抽样
    support_df, add_list = pd.DataFrame(), []
    label_number = len(label_list)
    count = [0] * label_number
    for label_i in range(label_number):
        # 取出当前标签的df
        label_i_df = df[df[label_list[label_i]] == 1]
        # 满足每个标签最少出现K次，如果原始数据集df不足K条则结束
        while count[label_i] < k and label_i_df.shape[0] > 0:
            add_series = label_i_df.sample(n=1)
            drop = label_i_df.drop(add_series.index, inplace=False)
            add_list.append(add_series)
            count = [i + j for i, j in zip(count, add_series['multi_label'].values[0])]
            df.drop(add_series.index, inplace=True)
            label_i_df = drop
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
    support_df = pd.DataFrame()
    query_df = pd.DataFrame()

    # 遍历选择的分类
    for category in selected_categories:
        # 获取该分类下的数据
        category_data = df[df[category] == 1]

        # 随机选择n个样本作为支持集
        support_samples = category_data.sample(n=shot)
        support_df = pd.concat([support_df, support_samples])

        # 从该分类中移除支持集样本，剩下的样本作为查询集
        category_data = category_data.drop(support_samples.index)

        # 随机选择n个样本作为查询集
        query_samples = category_data.sample(n=query)
        query_df = pd.concat([query_df, query_samples])

    print("Support Set and Query Set segmentation completed")
    return support_df, query_df


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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return dataloader


def get_dataloader_2(df, preprocess, label_list):
    data_x = df['cut_word'].values
    data_x = preprocess.get_pad_word2idx(data_x)
    data_x = [torch.tensor(i) for i in data_x.tolist()]

    # 创建输入数据的空列表
    label2id_list = []
    # 遍历数据集的每一行
    for index, row in df.iterrows():
        # 处理类别
        labels_tensor = torch.tensor([row[label] for label in label_list])
        label2id_list.append(labels_tensor)

    dataset = TensorDataset(torch.stack(data_x), torch.stack(label2id_list), torch.stack(label2id_list))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
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
            max_length=24,
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


def accuracy(y_true, y_pred):
    # 使用0.5作为阈值，大于阈值的为预测为正类
    y_pred = (y_pred < 0.5).float()
    # acc = (predicted_labels == y_true).float().mean()
    # 真实标签为[0, 1, 0, 1]，预测标签为[0, 1, 1, 0],acc = 1 / 3
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(torch.logical_and(y_true[i], y_pred[i]))
        q = sum(torch.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]


# 精确率
def Precision(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        count += sum(torch.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return count / y_true.shape[0]


# 召回率
def Recall(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        count += sum(torch.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return count / y_true.shape[0]


# F1
def F1Measure(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        p = sum(torch.logical_and(y_true[i], y_pred[i]))
        q = sum(y_true[i]) + sum(y_pred[i])
        count += (2 * p) / q
    return count / y_true.shape[0]


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
    criterion = nn.MultiLabelSoftMarginLoss(reduction='sum')
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
    model.train()
    epoch_los, epoch_acc = 0.0, 0.0
    for i, (support_input, query_input) in enumerate(zip(support_loader, query_loader)):
        # 1. 放到GPU上
        support_input0 = support_input[0].to(device, dtype=torch.long)
        support_input2 = support_input[2].to(device, dtype=torch.long)
        query_input0 = query_input[0].to(device, dtype=torch.long)
        query_input2 = query_input[2].to(device, dtype=torch.long)
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        output = model(support_input0, support_input2, query_input0)
        # outputs = outputs.squeeze(1)
        # 4. 计算损失
        loss = criterion(output, query_input2.float())
        # loss = multilabel_categorical_crossentropy(output, query_input2.float())
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
    return acc_value, loss_value


def evaluating(support_loader, test_loader, model):
    criterion = nn.MultiLabelSoftMarginLoss(reduction='sum')
    model.eval()
    with torch.no_grad():
        epoch_los, epoch_acc = 0.0, 0.0
        for i, (support_input, test_input) in enumerate(zip(support_loader, test_loader)):
            # 1. 放到GPU上
            support_input0 = support_input[0].to(device, dtype=torch.long)
            support_input2 = support_input[2].to(device, dtype=torch.long)
            query_input0 = test_input[0].to(device, dtype=torch.long)
            query_input2 = test_input[2].to(device, dtype=torch.long)
            # 2. 计算输出
            output = model(support_input0, support_input2, query_input0)
            # outputs = outputs.squeeze(1)
            # 3. 计算损失
            loss = criterion(output, query_input2.float())
            # loss = torch.sum(output, dim=0)
            epoch_los += loss.item()
            # 4.预测结果
            accu = accuracy(output, query_input2)
            epoch_acc += accu.item()
        loss_value = epoch_los / len(support_loader)
        acc_value = epoch_acc / len(support_loader)
    return acc_value, loss_value


# bert模型
def run_proto_bert():
    features = ['name', 'storeType']
    labels = ['碳酸饮料', '果汁', '茶饮', '水', '乳制品', '植物蛋白饮料', '功能饮料']
    # labels = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
    #           '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
    columns = ['store_id', 'drinkTypes']
    columns.extend(features)
    columns.extend(labels)

    labeled_df = pd.read_csv(labeled_path, usecols=columns)
    labeled_df = labeled_df[labeled_df['name'].notnull() & (labeled_df['name'] != '')]
    labeled_df = labeled_df[labeled_df['storeType'].notnull() & (labeled_df['storeType'] != '')]
    # # bert_config = AutoConfig.from_pretrained(pretrian_bert_url + '/config.json')

    tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)
    bert_layer = AutoModel.from_pretrained(pretrian_bert_url)
    proto_model = ProtoTypicalNet(
        bert_layer=bert_layer,
        input_dim=768,
        hidden_dim=128,
        num_class=len(labels)
    ).to(device)

    # # 采用NwayKshot采样
    # support_df, query_df = get_Nway_Kshot(labeled_df, labels, 7, 32, 8)

    # 采用最小包含算法采样
    train_set, test_set = train_test_split(labeled_df, test_size=0.2)
    print('train_set len:{} test_set len:{}'.format(train_set.shape[0], test_set.shape[0]))
    support_set = get_Support_Query(train_set, labels, k=200)
    query_set = train_set.drop(support_set.index)
    # support_set, query_set = get_Nway_Kshot(train_set, labels, 7, 64, 16)
    print('support_set len:{} query_set len:{}'.format(support_set.shape[0], query_set.shape[0]))

    # dataloader
    support_dataloader = get_labeled_dataloader(support_set, tokenizer, labels)
    query_dataloader = get_labeled_dataloader(query_set, tokenizer, labels)
    test_dataloader = get_labeled_dataloader(test_set, tokenizer, labels)
    # 训练 测试 分析
    for step in range(epochs):
        train_acc_value, train_loss_value = training(support_dataloader, query_dataloader, proto_model)
        test_acc_value, test_loss_value = evaluating(support_dataloader, test_dataloader, proto_model)
        print("epochs:{} 训练集 accuracy: {:.2%},loss:{:.4f} "
              "| 验证集 accuracy: {:.2%},loss:{:.4f}".format(step, train_acc_value, train_loss_value, test_acc_value,
                                                             test_loss_value))
        writer.add_scalars('acc', {'train_acc': train_acc_value, 'test_acc': test_acc_value}, global_step=step)
        writer.add_scalars('loss', {'train_loss': train_loss_value, 'test_loss': test_loss_value}, global_step=step)
    # get_unlabeled_dataloader(unlabeled_path, tokenizer)


# w2v模型
def run_proto_w2v():
    features = ['name', 'storeType']
    labels = ['碳酸饮料', '果汁', '茶饮', '水', '乳制品', '植物蛋白饮料', '功能饮料']
    labels = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
              '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
    columns = ['store_id', 'drinkTypes']
    columns.extend(features)
    columns.extend(labels)

    labeled_df = pd.read_csv(labeled_di_sku_path, usecols=columns)
    labeled_df = labeled_df[labeled_df['name'].notnull() & (labeled_df['name'] != '')]
    labeled_df = labeled_df[labeled_df['storeType'].notnull() & (labeled_df['storeType'] != '')]

    # 加载 data
    segment = WordSegment()
    labeled_df['cut_word'] = (labeled_df['name'] + labeled_df['storeType']).apply(segment.cut_word)
    data_x = labeled_df['cut_word'].values
    preprocess = Preprocess(sen_len=16)
    embedding = preprocess.create_tokenizer()

    # 采用最小包含算法采样
    train_set, test_set = train_test_split(labeled_df, test_size=0.2)
    print('train_set len:{} test_set len:{}'.format(train_set.shape[0], test_set.shape[0]))
    support_set = get_Support_Query(train_set, labels, k=200)
    query_set = train_set.drop(support_set.index)
    # support_set, query_set = get_Nway_Kshot(train_set, labels, 7, 64, 16)
    print('support_set len:{} query_set len:{}'.format(support_set.shape[0], query_set.shape[0]))

    # dataloader
    support_dataloader = get_dataloader_2(support_set, preprocess, labels)
    query_dataloader = get_dataloader_2(query_set, preprocess, labels)
    test_dataloader = get_dataloader_2(test_set, preprocess, labels)

    proto_model_2 = ProtoTypicalNet2(
        embedding=embedding,
        embedding_dim=200,
        hidden_dim=128,
        num_class=len(labels)
    ).to(device)

    # 训练 测试 分析
    for step in range(epochs):
        train_acc_value, train_loss_value = training(support_dataloader, query_dataloader, proto_model_2)
        test_acc_value, test_loss_value = evaluating(support_dataloader, test_dataloader, proto_model_2)
        print("epochs:{} 训练集 accuracy: {:.2%},loss:{:.4f} "
              "| 验证集 accuracy: {:.2%},loss:{:.4f}".format(step, train_acc_value, train_loss_value, test_acc_value,
                                                             test_loss_value))
        writer.add_scalars('acc', {'train_acc': train_acc_value, 'test_acc': test_acc_value}, global_step=step)
        writer.add_scalars('loss', {'train_loss': train_loss_value, 'test_loss': test_loss_value}, global_step=step)


if __name__ == '__main__':
    # run_proto_bert()
    #
    run_proto_w2v()

# tensorboard --logdir=E:\pyProjects\pycharm_project\workplace\fewsamples\logs\v1 --port 8123
