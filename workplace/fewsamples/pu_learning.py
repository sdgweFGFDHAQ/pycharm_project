import numpy
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

from workplace.fewsamples.models.proto_model_2 import ProtoTypicalNet2
from workplace.fewsamples.preprocess_data import Preprocess
from workplace.fewsamples.utils.mini_tool import WordSegment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
PU-Learning，即P代表的是Positive，U代表的是Unlabel，负样本实际上是泛样本
"""
noEDA_prefix = '/home/data/temp/lxb/alchemy/data/noEDA_data'
input_prefix = '/home/data/temp/lxb/alchemy/data/input_dataset'
labeled_path_update = './data/is_7t1.csv'
labeled_path = './data/di_sku_log_single_drink_labels.csv'

threshold = 5
epoch = 10
batch_size = 32


def difine_dataset(df, tokenizer, label_list):
    data_x = df['cut_word'].values
    data_x = tokenizer.get_pad_word2idx(data_x)
    data_x = [torch.tensor(i) for i in data_x.tolist()]

    # 创建输入数据的空列表
    label2id_list = []
    # 遍历数据集的每一行
    for index, row in df.iterrows():
        # 处理类别
        labels_tensor = torch.tensor([row[label] for label in label_list])
        label2id_list.append(labels_tensor)

    dataset = TensorDataset(torch.stack(data_x), torch.stack(label2id_list), torch.stack(df.index))
    return dataset


def accuracy(pred_y, y):
    pred_list = torch.argmax(pred_y, dim=1)
    correct = (pred_list == y).float()
    acc = correct.sum() / len(correct)
    return acc


def training(dataset, model):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
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
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
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
            score += output
        return score


def rerun():
    # 设置所需字段
    features = ['name', 'storeType']
    labels = ['碳酸饮料', '果汁', '茶饮', '水', '乳制品', '植物蛋白饮料', '功能饮料']
    labels = ['碳酸饮料']  # 先测试单标签
    columns = ['store_id']
    columns.extend(features)
    columns.extend(labels)
    # 读取文件数据
    labeled_df = pd.read_csv(labeled_path_update, usecols=columns)
    labeled_df = labeled_df[labeled_df['name'].notnull() & (labeled_df['name'] != '')]
    labeled_df = labeled_df[labeled_df['storeType'].notnull() & (labeled_df['storeType'] != '')]
    segment = WordSegment()
    labeled_df['cut_word'] = (labeled_df['name'] + labeled_df['storeType']).apply(segment.cut_word)

    # 划分正样本和无标签数据
    pos_index = labeled_df[labeled_df['碳酸饮料'] == 1].index
    unl_index = labeled_df[labeled_df['碳酸饮料'] == 0].index
    # 加载token、embedding
    tokenizer = Preprocess(sen_len=6)
    embedding = tokenizer.create_tokenizer()

    score_sum = pd.DataFrame(np.zeros(shape=len(unl_index)), index=unl_index)
    score_num = pd.DataFrame(np.zeros(shape=len(unl_index)), index=unl_index)
    condition = 0
    model = ProtoTypicalNet2(
        embedding=embedding,
        embedding_dim=200,
        hidden_dim=64,
        num_labels=len(labels)
    ).to(device)
    while condition < threshold:
        neg_index_temp = np.random.choice(unl_index, replace=True, size=len(pos_index))
        remaining_index = list(set(unl_index) - set(neg_index_temp))
        train_dataset = difine_dataset(labeled_df.loc[pos_index + neg_index_temp], tokenizer, labels)
        eval_dataset = difine_dataset(labeled_df.loc[remaining_index], tokenizer, labels)
        for _ in range(epoch):
            # (1)使用正样本和泛洋本训练分类器
            train_acc_value, train_loss_value = training(train_dataset, model)
            # (2)对泛样本打分，选取概率最高的作为负样本，重新生成样本集csv文件
            eval_acc_value, eval_loss_value = evaluating(eval_dataset, model)
            print("epochs:{} 训练集 accuracy: {:.2%},loss:{:.4f} "
                  "| 验证集 accuracy: {:.2%},loss:{:.4f}"
                  .format(epoch, train_acc_value, train_loss_value, eval_acc_value, eval_loss_value))
        score = predicting(eval_dataset, model)
        # (4)最终我们拿到了N次的预测结果，取平均作为最终的预测概率
        score_sum.loc[remaining_index, 0] += score
        score_num.loc[remaining_index, 0] += 1
    mean_scores = score_sum / score_num
    results = pd.concat([mean_scores, labeled_df[labeled_df['碳酸饮料'] == 0]], axis=1)
    results.sort_values(by='score', ascending=False).to_csv('pl_carbon.csv')


if __name__ == '__main__':
    rerun()
