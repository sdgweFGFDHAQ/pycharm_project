import random
import warnings

import numpy as np
import pandas as pd
from icecream.icecream import ic
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter('./logs/v2')

labeled_path = '../sv_report_data.csv'
labeled_update_path = './data/is_7t1.csv'
labeled_di_sku_path = './data/di_sku_log_drink_labels.csv'
pretrian_bert_url = "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese"

token_max_length = 12
batch_size = 64
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


# 获取数据集
def set_Nway_Kshot(df, category_list, shot, query):
    # 创建一个空的DataFrame用于存储支持集和查询集
    support_df = pd.DataFrame()
    query_df = pd.DataFrame()

    # 遍历选择的分类
    for category in category_list:
        # 获取该分类下的数据
        category_data = df[df[category] == 1]

        # 随机选择n个样本作为支持集
        support_samples = category_data.sample(n=shot)
        support_df = pd.concat([support_df, support_samples])

        # 从该分类中移除支持集样本，剩下的样本作为查询集
        category_data = category_data.drop(support_samples.index)
        # 随机选择1个样本作为查询集
        query_samples = category_data.sample(n=query)
        query_df = pd.concat([query_df, query_samples])

    print("Support Set and Query Set segmentation completed")
    return support_df, query_df


def define_dataloader(df, preprocess):
    data_x = df['cut_word'].values
    data_x = preprocess.get_pad_word2idx(data_x)
    data_x = [torch.tensor(i) for i in data_x.tolist()]
    dataset = TensorDataset(torch.stack(data_x))
    return dataset


def define_dataloader_2(df, preprocess, label_list):
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

    dataset = TensorDataset(torch.stack(data_x), torch.stack(label2id_list))
    return dataset


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


# 评价指标
def threshold_EVA(y_pred, y_true, rs):
    acc, pre, rec, f1 = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
    # 设置阈值
    # rs = torch.mean(y_true.float(), dim=0)
    # max_values, min_values = torch.max(y_pred, dim=0).values, torch.min(y_pred, dim=0).values
    # thresholds = (max_values - min_values) * rs + min_values
    # y_pred = torch.where(y_pred >= thresholds, torch.ones_like(y_pred), torch.zeros_like(y_pred))
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


# 训练
def training(dataset, model, r_list):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.BCEWithLogitsLoss(weight=r_list, reduction='sum')
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    model.train()
    epoch_los, epoch_acc, epoch_prec, epoch_recall, epoch_f1s = 0.0, 0.0, 0.0, 0.0, 0.0
    for i, support_input in enumerate(dataloader):
        # 1. 放到GPU上
        support_input0 = support_input[0].to(device, dtype=torch.long)
        support_input2 = support_input[1].to(device, dtype=torch.long)
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        output = model(support_input0)
        # 4. 计算损失
        loss = criterion(output, support_input2.float())
        # 5. 反向传播
        loss.requires_grad_(True)
        loss.backward()
        # 6. 更新梯度
        optimizer.step()
        # 7.预测结果
        epoch_los += loss.item()
        accu, precision, recall, f1s = threshold_EVA(output, support_input2, r_list)
        epoch_acc += accu.item()
        epoch_prec += precision.item()
        epoch_recall += recall.item()
        epoch_f1s += f1s.item()

    loss_value = epoch_los / len(dataloader)
    acc_value = epoch_acc / len(dataloader)
    prec_value = epoch_prec / len(dataloader)
    rec_value = epoch_recall / len(dataloader)
    f1_value = epoch_f1s / len(dataloader)
    return acc_value, loss_value, prec_value, rec_value, f1_value


# 测试
def evaluating(dataset, model, r_list):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    criterion = nn.BCEWithLogitsLoss(weight=r_list, reduction='sum')

    model.eval()
    with torch.no_grad():
        epoch_los, epoch_acc, epoch_prec, epoch_recall, epoch_f1s = 0.0, 0.0, 0.0, 0.0, 0.0
        for i, support_input in enumerate(dataloader):
            # 1. 放到GPU上
            support_input0 = support_input[0].to(device, dtype=torch.long)
            support_input2 = support_input[1].to(device, dtype=torch.long)
            # 2. 计算输出
            output = model(support_input0)
            # 3. 计算损失
            loss = criterion(output, support_input2.float())
            epoch_los += loss.item()
            # 4.预测结果
            accu, precision, recall, f1s = threshold_EVA(output, support_input2, r_list)
            epoch_acc += accu.item()
            epoch_prec += precision.item()
            epoch_recall += recall.item()
            epoch_f1s += f1s.item()
        loss_value = epoch_los / len(dataloader)
        acc_value = epoch_acc / len(dataloader)
        prec_value = epoch_prec / len(dataloader)
        rec_value = epoch_recall / len(dataloader)
        f1_value = epoch_f1s / len(dataloader)
    return acc_value, loss_value, prec_value, rec_value, f1_value


# 预测
def predicting(dataset, model, r_list):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model.eval()
    with torch.no_grad():
        label_list = []
        for i, support_input in enumerate(dataloader):
            # 1. 放到GPU上
            support_input0 = support_input[0].to(device, dtype=torch.long)
            # 2. 计算输出
            output = model(support_input0)
            label_list.extend([tensor.numpy() for tensor in output])
        # max_values, min_values = np.max(label_list, axis=0), np.min(label_list, axis=0)
        # thresholds = (max_values - min_values) * r_list.numpy() + min_values
        # label_list = np.where(label_list > thresholds, 1, 0)
        label_list = [[np.where(arr > 0.5, 1, 0) for arr in row] for row in label_list]
    return label_list


# 训练 测试 分析
def train_and_test(support_dataset, test_dataset, proto_model_2, ratio, save_path):
    max_accuracy = 0.0
    for step in range(epochs):
        train_acc_value, train_loss_value, train_prec_value, \
            train_rec_value, train_f1_value = training(support_dataset, proto_model_2, ratio)
        test_acc_value, test_loss_value, test_prec_value, \
            test_rec_value, test_f1_value = evaluating(test_dataset, proto_model_2, ratio)
        print("epochs:{} 训练集 accuracy: {:.2%},loss:{:.4f} "
              "| 验证集 accuracy: {:.2%},loss:{:.4f}"
              .format(step, train_acc_value, train_loss_value, test_acc_value, test_loss_value))
        print("         -- 训练集 precision: {:.2%},recall: {:.2%},F1:{:.2%} "
              "| 验证集 precision: {:.2%},recall: {:.2%},F1:{:.2%}"
              .format(train_prec_value, train_rec_value, train_f1_value, test_prec_value, test_rec_value,
                      test_f1_value))
        # writer.add_scalars('acc', {'train_acc': train_acc_value, 'test_acc': test_acc_value}, global_step=step)
        # writer.add_scalars('loss', {'train_loss': train_loss_value, 'test_loss': test_loss_value}, global_step=step)

        # 保存最佳模型
        if test_acc_value > max_accuracy:
            max_accuracy = test_acc_value
            torch.save(proto_model_2.state_dict(), save_path)


# w2v模型
def run_proto_w2v():
    features = ['name', 'storeType']
    labels = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
              '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
    columns = ['drinkTypes']
    columns.extend(features)
    columns.extend(labels)

    # 读取指定列，去除空值
    labeled_df = pd.read_csv(labeled_di_sku_path, usecols=columns)
    labeled_df = labeled_df[labeled_df['name'].notnull() & (labeled_df['name'] != '')]
    labeled_df = labeled_df[labeled_df['storeType'].notnull() & (labeled_df['storeType'] != '')]
    # 清洗中文文本
    segment = WordSegment()
    labeled_df['cut_word'] = (labeled_df['name'] + labeled_df['storeType']).apply(segment.cut_word)
    preprocess = Preprocess(sen_len=5)
    embedding = preprocess.create_tokenizer()

    # 采用最小包含算法采样
    sq_set = get_Support_Query(labeled_df, labels, k=2000)
    # sq_set.to_csv('./data/test_support_set3.csv', index=False)
    print('sq_set len:{}'.format(sq_set.shape[0]))
    test_set = labeled_df.drop(sq_set.index)
    # sq_set.to_csv('./data/test_support_set3.csv', index=False)
    # support_set, query_set = train_test_split(sq_set, test_size=0.2)
    # print('train_set len:{} test_set len:{}'.format(train_set.shape[0], test_set.shape[0]))

    # dataloader
    support_dataset = define_dataloader_2(sq_set, preprocess, labels)
    test_dataset = define_dataloader_2(test_set, preprocess, labels)

    # 计算标签为0的占比,作为阈值
    num_ones = torch.tensor((sq_set[labels] == 1).sum(axis=0))
    ratio = num_ones / sq_set.shape[0]

    proto_model_2 = ProtoTypicalNet2(
        embedding=embedding,
        embedding_dim=200,
        hidden_dim=128,
        num_labels=len(labels)
    ).to(device)
    # 训练 测试 分析
    train_and_test(support_dataset, test_dataset, proto_model_2, ratio, './models/proto_model_3.pth')

    print("=================================")

    # 加载模型做预测
    proto_model_2 = ProtoTypicalNet2(
        embedding=embedding,
        embedding_dim=200,
        hidden_dim=128,
        num_labels=len(labels)
    ).to(device)

    proto_model_2.load_state_dict(torch.load('./models/proto_model_3.pth'))
    lable_result = predicting(test_dataset, proto_model_2, ratio)
    drink_df = pd.DataFrame(lable_result, columns=labels)
    source_df = test_set[['name', 'storeType', 'drinkTypes']].reset_index(drop=True)
    predict_result = pd.concat([source_df, drink_df], axis=1)
    predict_result.to_csv('./data/sku_predict_result3.csv')


if __name__ == '__main__':
    run_proto_w2v()
# tensorboard --logdir=E:\pyProjects\pycharm_project\workplace\fewsamples\logs\v1 --port 8123
