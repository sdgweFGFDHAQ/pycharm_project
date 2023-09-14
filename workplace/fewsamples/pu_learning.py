from icecream import ic
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

from models.proto_model_2 import ProtoTypicalNet2, ProtoTypicalNet3
from preprocess_data import Preprocess
from utils.mini_tool import WordSegment, Logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = Logger()
"""
PU-Learning，即P代表的是Positive，U代表的是Unlabel，负样本实际上是泛样本
"""
labeled_path_update = './data/is_7t1.csv'
labeled_di_sku_path = './data/di_sku_log_single_drink_labels.csv'
labeled_di_sku_path2 = './data/di_sku_log_chain_data.csv'
threshold = 5
epoch = 20
batch_size = 64


def difine_dataset(df, tokenizer, label_list):
    data_x = df['cut_word'].values
    data_x = tokenizer.get_pad_word2idx(data_x)
    data_x = [torch.tensor(i) for i in data_x.tolist()]
    df_index = [torch.tensor(i) for i in df.index]

    # 创建输入数据的空列表
    label2id_list = []
    # 遍历数据集的每一行
    for index, row in df.iterrows():
        # 处理类别
        labels_tensor = torch.tensor([row[label] for label in label_list])
        label2id_list.append(labels_tensor)

    dataset = TensorDataset(torch.stack(data_x), torch.stack(label2id_list), torch.stack(df_index))
    return dataset


def accuracy(pred_y, y):
    pred_y = (pred_y > 0.5).int()

    correct = (pred_y == y).int()
    acc = correct.sum() / len(correct)
    return acc


def training(dataset, model):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    model.train()
    epoch_los, epoch_acc = 0.0, 0.0
    for i, support_input in enumerate(dataloader):
        feature = support_input[0].to(device, dtype=torch.long)
        label = support_input[1].to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(feature)
        loss = criterion(output, label.float())
        loss.backward()
        optimizer.step()
        epoch_los += loss.item()
        accu = accuracy(output, label)
        epoch_acc += accu.item()
    num_batches = len(dataloader)
    loss_value = epoch_los / num_batches
    acc_value = epoch_acc / num_batches
    return acc_value, loss_value


def evaluating(dataset, model):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    model.eval()
    epoch_los, epoch_acc, score = 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for i, support_input in enumerate(dataloader):
            feature = support_input[0].to(device, dtype=torch.long)
            label = support_input[1].to(device, dtype=torch.long)
            output = model(feature)
            loss = criterion(output, label.float())
            epoch_los += loss.item()
            accu = accuracy(output, label)
            epoch_acc += accu.item()
    num_batches = len(dataloader)
    loss_value = epoch_los / num_batches
    acc_value = epoch_acc / num_batches
    return acc_value, loss_value


def predicting(dataset, model):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    score = []
    with torch.no_grad():
        for i, support_input in enumerate(dataloader):
            feature = support_input[0].to(device, dtype=torch.long)
            output = model(feature)
            score += output.cpu().tolist()
        return score


def rerun():
    # 设置所需字段
    features = ['name', 'storetype']
    labels = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
              '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
    labels = ['碳酸饮料']  # 先测试单标签
    columns = []
    columns.extend(features)
    columns.extend(labels)

    # 读取文件数据
    labeled_df = pd.read_csv(labeled_di_sku_path, usecols=columns)
    labeled_df = labeled_df[labeled_df['name'].notnull() & (labeled_df['name'] != '')]
    labeled_df = labeled_df[labeled_df['storetype'].notnull() & (labeled_df['storetype'] != '')]
    labeled_df.reset_index(inplace=True)
    segment = WordSegment()
    labeled_df['cut_word'] = (labeled_df['name'] + labeled_df['storetype']).apply(segment.cut_word)

    # 划分正样本和无标签数据
    pos_index = labeled_df[labeled_df['碳酸饮料'] == 1].index
    unl_index = labeled_df[labeled_df['碳酸饮料'] == 0].index
    print("正样本数量：{0} 负样本数量：{1}".format(len(pos_index), len(unl_index)))
    # 加载token、embedding
    tokenizer = Preprocess(sen_len=6)
    embedding = tokenizer.create_tokenizer()

    score_sum = np.zeros(shape=(len(labeled_df.index), 1))
    score_num = np.zeros(shape=(len(labeled_df.index), 1))
    condition = 0

    while condition < threshold:
        # 操作次数计数
        condition += 1
        # 初始化分类器
        model = ProtoTypicalNet3(
            embedding=embedding,
            embedding_dim=200,
            hidden_dim=64,
            num_labels=len(labels)
        ).to(device)

        print("=========第{}次采样============".format(condition))
        # 从无标签样本中抽一部分作为负样本
        neg_index_temp = np.random.choice(unl_index, replace=True, size=len(pos_index))
        # 无标签样本中剩下的部分
        remaining_index = list(set(unl_index) - set(neg_index_temp))
        # 正负样本构建训练集
        train_dataset = difine_dataset(labeled_df.loc[list(set(pos_index) | set(neg_index_temp))], tokenizer, labels)
        # 剩下的作为验证集
        eval_dataset = difine_dataset(labeled_df.loc[remaining_index], tokenizer, labels)
        # 训练epoch次分类器
        for i in range(epoch):
            # (1)使用正样本和泛洋本训练分类器
            train_acc_value, train_loss_value = training(train_dataset, model)
            eval_acc_value, eval_loss_value = evaluating(eval_dataset, model)
            print("epochs:{} 训练集 accuracy: {:.2%},loss:{:.4f} "
                  "| 验证集 accuracy: {:.2%},loss:{:.4f}"
                  .format(i, train_acc_value, train_loss_value, eval_acc_value, eval_loss_value))
        # (2)对泛样本打分，选取概率最高的作为负样本，重新生成样本集csv文件
        score = predicting(eval_dataset, model)
        score_sum[remaining_index, :] += score
        score_num[remaining_index, :] += 1
    score_num = np.where(score_num == 0, 1, score_num)
    mean_scores = score_sum / score_num
    results = pd.concat([labeled_df, pd.DataFrame(mean_scores, columns=['score'])], axis=1)
    results.sort_values(by='score', ascending=False).to_csv('pl_carbon.csv', index=False)


if __name__ == '__main__':
    rerun()
    # 取出负样本-正样本数量的高评分数据，加入正样本，使用原型网络训练
