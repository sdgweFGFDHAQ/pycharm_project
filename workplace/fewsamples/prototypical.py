import random

import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from models.proto_model import ProtoTypicalNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrian_bert_url = "hfl/chinese-roberta-wwm-ext"
labeled_path = '../sv_report_data.csv'
unlabeled_path = '../unlabeled_data.csv'

token_max_length = 12
batch_size = 64
epochs = 3


def get_Nway_Kshot(df, category_list, way, shot):
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
        query_samples = category_data.sample(n=shot)
        query_set = pd.concat([query_set, query_samples])

    print("Support Set and Query Set segmentation completed")
    return support_set, query_set


# 线下跑店(店铺存在且售品) 1399条
def get_labeled_dataloader(df, bert_tokenizer, label_list):
    # 生成类别-id字典
    df['cat_id'] = df['storeType'].factorize()[0]
    cat_df = df[['storeType', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
    cat_df.to_csv('./data/store_type_to_id.csv')

    # 创建输入数据的空列表
    input_ids = []
    attention_masks = []

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
        label_list.append(labels_tensor)
        print(label_list)
    dataset = TensorDataset(torch.stack(input_ids), torch.stack(attention_masks), torch.stack(label_list))
    dataloader = DataLoader(dataset, batch_size=batch_size)
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
    pred_list = torch.argmax(pred_y, dim=1)
    correct = (pred_list == y).float()
    acc = correct.sum() / len(correct)
    return acc


def training(support_loader, query_loader, query_label, model):
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)
    model.train()
    epoch_los, epoch_acc = 0.0, 0.0
    for i, (support_input, query_input) in enumerate(support_loader, query_loader):
        # 1. 放到GPU上
        support_input = support_input.to(device, dtype=torch.float32)
        query_input = query_input.to(device, dtype=torch.float32)
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        output = model(support_input, query_input)
        # outputs = outputs.squeeze(1)
        # 4. 计算损失
        loss = nn.NLLLoss(output, query_label)
        epoch_los += loss.item()
        # 5.预测结果
        accu = accuracy(output, query_label)
        epoch_acc += accu.item()
        # 6. 反向传播
        loss.backward()
        # 7. 更新梯度
        optimizer.step()
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
    # bert_config = AutoConfig.from_pretrained(pretrian_bert_url + '/config.json')
    tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)

    bert_layer = AutoModel.from_pretrained(pretrian_bert_url)
    proto_model = ProtoTypicalNet(
        bert_layer=bert_layer,
        input_dim=768,
        hidden_dim=768,
        num_class=len(labels)
    ).to(device)

    for i in range(epochs):
        support_df, query_df = get_Nway_Kshot(labeled_df, labels, 3, 64)
        support_dataloader = get_labeled_dataloader(support_df, tokenizer, labels)
        query_dataloader = get_labeled_dataloader(query_df, tokenizer, labels)
        training(support_dataloader, query_dataloader, labels, proto_model)

    # get_unlabeled_dataloader(unlabeled_path, tokenizer)
