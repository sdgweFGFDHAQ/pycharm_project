import warnings

import numpy as np
import pandas as pd
from icecream.icecream import ic
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter

from models.proto_model import ProtoTypicalNet

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./logs/v1')

pretrian_bert_url = "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese"

labeled_path = '../sv_report_data.csv'
labeled_update_path = './data/is_7t1.csv'
labeled_di_sku_path = './data/di_sku_log_drink_labels.csv'

token_max_length = 12
batch_size = 16
epochs = 15


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
            max_length=14,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'].squeeze())
        attention_masks.append(encoded_dict['attention_mask'].squeeze())

        # 处理类别
        labels_tensor = torch.tensor([row[label] for label in label_list])
        label2id_list.append(labels_tensor)

    dataset = TensorDataset(torch.stack(input_ids), torch.stack(label2id_list), torch.stack(attention_masks))
    return dataset


def threshold_EVA(y_pred, y_true):
    acc, pre, rec, f1 = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
    # 设置阈值
    y_pred = (y_pred > 0.5).int()  # 使用0.5作为阈值，大于阈值的为预测为正类
    try:
        # 准确率
        correct = (y_pred == y_true).int()
        acc = correct.sum() / (correct.shape[0] * correct.shape[1])
        # acc = accuracy_score(y_pred, y_true)
        # 精确率
        pre = precision_score(y_pred, y_true, average='weighted')
        # 召回率
        rec = recall_score(y_pred, y_true, average='weighted')
        # F1
        f1 = f1_score(y_pred, y_true, average='weighted')
    except Exception as e:
        print(str(e))
    return acc, pre, rec, f1


def training(support_set, model, r_list):
    support_loader = DataLoader(support_set, batch_size=batch_size, shuffle=False, drop_last=True)

    criterion = nn.BCEWithLogitsLoss(weight=r_list, reduction='sum')
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    model.train()
    epoch_los, epoch_acc, epoch_prec, epoch_recall, epoch_f1s = 0.0, 0.0, 0.0, 0.0, 0.0
    for i, support_input in enumerate(support_loader):
        # 1. 放到GPU上
        feature = support_input[0].to(device, dtype=torch.long)
        label = support_input[1].to(device, dtype=torch.long)
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        output = model(feature)
        # 4. 计算损失
        loss = criterion(output, label.float())
        epoch_los += loss.item()
        # 5.预测结果
        accu, precision, recall, f1s = threshold_EVA(output, label)
        epoch_acc += accu.item()
        epoch_prec += precision.item()
        epoch_recall += recall.item()
        epoch_f1s += f1s.item()
        # 6. 反向传播
        loss.requires_grad_(True)
        loss.backward()
        # 7. 更新梯度
        optimizer.step()
    num_batches = len(support_loader)
    loss_value = epoch_los / num_batches
    acc_value, prec_value = epoch_acc / num_batches, epoch_prec / num_batches
    rec_value, f1_value = epoch_recall / num_batches, epoch_f1s / num_batches
    return acc_value, loss_value, prec_value, rec_value, f1_value


def evaluating(support_set, model, r_list):
    support_loader = DataLoader(support_set, batch_size=batch_size, shuffle=False, drop_last=True)

    criterion = nn.BCEWithLogitsLoss(weight=r_list, reduction='sum')

    model.eval()
    with torch.no_grad():
        epoch_los, epoch_acc, epoch_prec, epoch_recall, epoch_f1s = 0.0, 0.0, 0.0, 0.0, 0.0
        for i, support_input in enumerate(support_loader):
            # 1. 放到GPU上
            feature = support_input[0].to(device, dtype=torch.long)
            label = support_input[1].to(device, dtype=torch.long)
            # 2. 计算输出
            output = model(feature)
            # 3. 计算损失
            loss = criterion(output, label.float())
            # loss = torch.sum(output, dim=0)
            epoch_los += loss.item()
            # 4.预测结果
            accu, precision, recall, f1s = threshold_EVA(output, label)
            epoch_acc += accu.item()
            epoch_prec += precision.item()
            epoch_recall += recall.item()
            epoch_f1s += f1s.item()
        num_batches = len(support_loader)
        loss_value = epoch_los / num_batches
        acc_value, prec_value = epoch_acc / num_batches, epoch_prec / num_batches
        rec_value, f1_value = epoch_recall / num_batches, epoch_f1s / num_batches
    return acc_value, loss_value, prec_value, rec_value, f1_value


def predicting(support_set, model, r_list):
    support_loader = DataLoader(support_set, batch_size=batch_size, shuffle=False, drop_last=True)

    model.eval()
    with torch.no_grad():
        label_list = []
        for i, support_input in enumerate(support_loader):
            # 1. 放到GPU上
            feature = support_input[0].to(device, dtype=torch.long)
            # 2. 计算输出
            output = model(feature)
            label_list.extend([tensor.numpy() for tensor in output])
        label_list = [[np.where(arr > 0.5, 1, 0) for arr in row] for row in label_list]
    return label_list


# 训练 测试 分析
def use_model(support_dataset, proto_model_2, ratio, save_path):
    max_accuracy = 0.0
    for step in range(epochs):
        train_acc_value, train_loss_value, train_prec_value, \
            train_rec_value, train_f1_value = training(support_dataset, proto_model_2, ratio)
        test_acc_value, test_loss_value, test_prec_value, \
            test_rec_value, test_f1_value = evaluating(support_dataset, proto_model_2, ratio)
        print("epochs:{} 训练集 accuracy: {:.2%},loss:{:.4f} "
              "| 验证集 accuracy: {:.2%},loss:{:.4f}"
              .format(step, train_acc_value, train_loss_value, test_acc_value, test_loss_value))
        print("         -- 训练集 precision: {:.2%},recall: {:.2%},F1:{:.2%} "
              "| 验证集 precision: {:.2%},recall: {:.2%},F1:{:.2%}"
              .format(train_prec_value, train_rec_value, train_f1_value, test_prec_value, test_rec_value,
                      test_f1_value))
        writer.add_scalars('acc', {'train_acc': train_acc_value, 'test_acc': test_acc_value}, global_step=step)
        writer.add_scalars('loss', {'train_loss': train_loss_value, 'test_loss': test_loss_value}, global_step=step)

        # 保存最佳模型
        if test_acc_value > max_accuracy:
            max_accuracy = test_acc_value
            torch.save(proto_model_2.state_dict(), save_path)


# 获取数据集
def get_dataset(labeled_df, labels):
    # 采用最小包含算法采样
    train_set, test_set = train_test_split(labeled_df, test_size=0.2)
    print('train_set len:{} test_set len:{}'.format(train_set.shape[0], test_set.shape[0]))
    support_set = get_Support_Query(train_set, labels, k=600)
    # train_set = train_set.drop(support_set.index)
    # query_set = get_Support_Query(train_set, labels, k=200)
    # print('support_set len:{} query_set len:{}'.format(support_set.shape[0], query_set.shape[0]))

    support_set.to_csv('./data/test_support_set3.csv', index=False)
    test_set.to_csv('./data/test_test_set3.csv', index=False)


# bert模型
def run_proto_bert():
    features = ['name', 'storeType']
    labels = ['碳酸饮料', '果汁', '茶饮', '水', '乳制品', '植物蛋白饮料', '功能饮料']
    columns = ['store_id', 'drinkTypes']
    columns.extend(features)
    columns.extend(labels)

    labeled_df = pd.read_csv(labeled_path, usecols=columns)
    labeled_df = labeled_df[labeled_df['name'].notnull() & (labeled_df['name'] != '')]
    labeled_df = labeled_df[labeled_df['storeType'].notnull() & (labeled_df['storeType'] != '')]

    # # 采用最小包含算法采样
    get_dataset(labeled_df, labels)
    support_set = pd.read_csv('./data/test_support_set.csv')
    test_set = pd.read_csv('./data/test_test_set.csv')

    # 计算标签为0的占比,作为阈值
    num_ones = torch.tensor((support_set[labels] == 1).sum(axis=0))
    ratio = num_ones / support_set.shape[0]

    tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)
    bert_layer = AutoModel.from_pretrained(pretrian_bert_url)
    proto_model = ProtoTypicalNet(
        bert_layer=bert_layer,
        input_dim=768,
        hidden_dim=128,
        num_class=len(labels)
    ).to(device)

    # dataloader
    support_dataset = get_labeled_dataloader(support_set, tokenizer, labels)
    # 训练 测试 分析
    use_model(support_dataset, proto_model, ratio, './models/proto_model.pth')

    # 加载模型做预测
    proto_model = ProtoTypicalNet(
        bert_layer=bert_layer,
        input_dim=768,
        hidden_dim=128,
        num_class=len(labels)
    ).to(device)

    proto_model.load_state_dict(torch.load('./models/proto_model.pth'))
    lable_result = predicting(support_dataset, proto_model, ratio)

    drink_df = pd.DataFrame(lable_result, columns=labels)
    predict_result = pd.concat([support_set[['store_id', 'name', 'storeType', 'drinkTypes']], drink_df], axis=1)
    predict_result.to_csv('./data/sku_predict_result.csv')


if __name__ == '__main__':
    run_proto_bert()
# tensorboard --logdir=E:\pyProjects\pycharm_project\workplace\fewsamples\logs\v1 --port 8123
