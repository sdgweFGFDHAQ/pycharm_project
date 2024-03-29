import os

from icecream import ic
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from workplace.fewsamples.preprocess_data import Preprocess
from workplace.fewsamples.models.model import LSTMNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 各数据集的路径
path_prefix = ''
train_with_label = os.path.join(path_prefix, './data/input_data.csv')
train_no_label = os.path.join(path_prefix, '')

# word2vec模型文件路径
w2v_path = os.path.join(path_prefix, 'models/word2vec.vector')

# 定义句子长度、是否固定 embedding、batch 大小、定义训练次数 epoch、learning rate 的值、model 的保存路径
model_dir = os.path.join(path_prefix, 'model/')
batch_size = 32
epochs = 5
lr = 0.001


class DefineDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def accuracy(pred_y, y):
    pred_list = torch.argmax(pred_y, dim=1)
    correct = (pred_list == y).float()
    acc = correct.sum() / len(correct)
    return acc


def training(train_loader, model):
    # 多分类损失函数
    criterion = nn.CrossEntropyLoss()
    # crit = nn.CrossEntropyLoss(reduction='sum')
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 將 model 的模式设定为 train，这样 optimizer 就可以更新 model 的参数
    model.train()
    train_len = len(train_loader)
    epoch_los, epoch_acc = 0, 0
    for i, (inputs, labels) in enumerate(train_loader):
        # 1. 放到GPU上
        inputs = inputs.to(device, dtype=torch.long)
        # inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        outputs = model(inputs)
        outputs = outputs.squeeze(1)  # 去掉最外面的 dimension
        # 4. 计算损失
        # outputs:batch_size*num_classes labels:1D
        loss = criterion(outputs, labels)
        # print(loss, outputs, labels)
        epoch_los += loss.item()
        # 5.预测结果
        accu = accuracy(outputs, labels)
        epoch_acc += accu.item()
        # 6. 反向传播
        loss.backward()
        # 7. 更新梯度
        optimizer.step()
    loss_value = epoch_los / train_len
    acc_value = epoch_acc / train_len * 100
    print('\nTrain | Loss:{:.5f} Acc: {:.3f}%'.format(loss_value, acc_value))
    return loss_value, acc_value


def evaluting(val_loader, model):
    # 多分类损失函数
    criterion = nn.CrossEntropyLoss()
    # crit = nn.CrossEntropyLoss(reduction='sum')
    # 將 model 的模式设定为 eval，固定model的参数
    model.eval()
    val_len = len(val_loader)
    with torch.no_grad():
        epoch_los, epoch_acc = 0, 0
        for i, (inputs, labels) in enumerate(val_loader):
            # 1. 放到GPU上
            inputs = inputs.to(device, dtype=torch.long)
            # inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            # 2. 计算输出
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            # 3. 计算损失
            loss = criterion(outputs, labels)
            epoch_los += loss.item()
            # 4. 预测结果
            accu = accuracy(outputs, labels)
            epoch_acc += accu.item()
        loss_value = epoch_los / val_len
        acc_value = epoch_acc / val_len * 100
        print("Valid | Loss:{:.5f} Acc: {:.3f}% ".format(loss_value, acc_value))
    print('-----------------------------------')
    return loss_value, acc_value


def load_programs(train_path):
    # 加载 data
    input_df = pd.read_csv(train_path)
    data_x, data_y = input_df['cut_name'].values, input_df['category3_new'].values
    category_classes = input_df['category3_new'].unique()
    # data pre_processing
    preprocess = Preprocess(sen_len=7)
    # 设置sen_len
    preprocess.length_distribution(data_x)
    # 加载model paragram
    embedding = preprocess.create_tokenizer()
    # 初始化参数
    data_x = preprocess.get_pad_word2idx(data_x)
    data_y = preprocess.get_lab2idx(data_y)

    return data_x, data_y, embedding, len(category_classes)


# class ContrastiveLoss(torch.nn.Module):
#     def __init__(self, margin=2.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         # calmp夹断用法
#         loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#         return loss_contrastive


def search_best_dataset(data_x, data_y, embedding, category_count):
    # 使用k折交叉验证
    best_x_train, best_y_train, best_x_test, best_y_test = None, None, None, None
    kf_5 = KFold(n_splits=10)
    k = 0
    best_accuracy = 0.
    for t_train, t_test in kf_5.split(data_x, data_y):
        print('==================第{}折================'.format(k + 1))
        k += 1
        model = LSTMNet(
            embedding,
            embedding_dim=200,
            hidden_dim=128,
            num_classes=category_count,
            num_layers=2,
            dropout=0.5,
            requires_grad=False
        ).to(device)
        train_ds = DefineDataset(data_x[t_train], data_y[t_train])
        train_ip = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        test_ds = DefineDataset(data_x[t_test], data_y[t_test])
        test_ip = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
        accuracy_list = list()
        # run epochs
        for ep in range(epochs):
            training(train_ip, model)
            _, ep_percent = evaluting(test_ip, model)
            accuracy_list.append(round(ep_percent, 3))
        mean_accuracy = np.mean(accuracy_list)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_x_train, best_y_train = data_x[t_train], data_y[t_train]
            best_x_test, best_y_test = data_x[t_test], data_y[t_test]
        print('Mean-Accuracy: {:.3f}'.format(mean_accuracy))
    print('Best model with acc {:.3f}%'.format(best_accuracy))
    return best_x_train, best_x_test, best_y_train, best_y_test


def draw_trend(train_ll, train_al, test_ll, test_al):
    # 绘制损失函数趋势图
    plt.title('Loss')
    plt.plot(train_ll, label='train')
    plt.plot(test_ll, label='test')
    plt.legend()
    plt.show()
    # 绘制准确率趋势图
    plt.title('Accuracy')
    plt.plot(train_al, label='train')
    plt.plot(test_al, label='test')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # train_with_label = './data/few_shot.csv'
    d_x, d_y, embedding_matrix, category_cou = load_programs(train_with_label)
    # imbalance = BorderlineSMOTE()
    # x_smote, y_smote = imbalance.fit_resample(d_x, d_y)
    # print(len(x_smote))
    lstm_model = LSTMNet(
        embedding_matrix,
        embedding_dim=200,
        hidden_dim=128,
        num_classes=category_cou,
        num_layers=2,
        dropout=0.5,
        requires_grad=False
    ).to(device)
    # 返回model中的参数的总数目
    total = sum(p.numel() for p in lstm_model.parameters())
    trainable = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    # K折交叉验证
    x_train, x_test, y_train, y_test = search_best_dataset(d_x, d_y, embedding_matrix, category_cou)
    # split data
    # x_train, x_test, y_train, y_test = train_test_split(d_x, d_y, test_size=0.3, random_state=5)

    # 构造Dataset
    train_dataset = DefineDataset(x_train, y_train)
    val_dataset = DefineDataset(x_test, y_test)
    # preparing the training loader
    train_input = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print('Training loader prepared.')
    # preparing the validation loader
    val_input = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print('Validation loader prepared.')

    best_acc = 0.
    epochs = 20
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = list(), list(), list(), list()
    # run epochs
    for epoch in range(epochs):
        print('[ Epoch{}: batch_size({}) ]'.format(epoch + 1, batch_size))
        # train for one epoch
        epoch_loss, epoch_accuracy = training(train_input, lstm_model)
        # predict on validation set
        epoch_distance, epoch_percent = evaluting(val_input, lstm_model)
        if epoch_percent > best_acc:
            # 如果 validation 的结果好于之前所有的结果，就把当下的模型保存
            best_acc = epoch_percent
            # torch.save(model, "{}/ckpt.model".format(model_dir))
            print('saving model with acc {:.3f}%'.format(epoch_percent))

        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_accuracy)
        test_loss_list.append(epoch_distance)
        test_acc_list.append(epoch_percent)

    # 绘制趋势图像
    draw_trend(train_loss_list, train_acc_list, test_loss_list, test_acc_list)
    print('训练集平均损失值： {0}, 平均准确率{1}'.format(np.mean(train_loss_list), np.mean(train_acc_list)))
    print('测试集平均损失值： {0}, 平均测试率{1}'.format(np.mean(test_loss_list), np.mean(test_acc_list)))
