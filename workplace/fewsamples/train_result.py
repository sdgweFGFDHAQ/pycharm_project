import os

from icecream import ic
import numpy as np
import pandas as pd
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
epochs = 10
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
    epoch_loss, epoch_acc = 0, 0
    for i, (inputs, labels) in enumerate(train_loader):
        # 1. 放到GPU上
        inputs = inputs.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.long)
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        outputs = model(inputs)
        outputs = outputs.squeeze(1)  # 去掉最外面的 dimension
        # 4. 计算损失
        # outputs:batch_size*num_classes labels:1D
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        # 5.预测结果
        accu = accuracy(outputs, labels)
        epoch_acc += accu.item()
        # 6. 反向传播
        loss.backward()
        # 7. 更新梯度
        optimizer.step()
    ic(epoch_acc)
    print('\nTrain | Loss:{:.5f} Acc: {:.3f}%'.format(epoch_loss / train_len, epoch_acc / train_len * 100))


def predicting(val_loader, model):
    # 多分类损失函数
    criterion = nn.CrossEntropyLoss()
    # crit = nn.CrossEntropyLoss(reduction='sum')
    # 將 model 的模式设定为 eval，固定model的参数
    model.eval()
    val_len = len(val_loader)
    with torch.no_grad():
        epoch_loss, epoch_acc = 0, 0
        for i, (inputs, labels) in enumerate(val_loader):
            # 1. 放到GPU上
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            # 2. 计算输出
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            # 3. 计算损失
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            # 4. 预测结果
            accu = accuracy(outputs, labels)
            epoch_acc += accu.item()
        print("Valid | Loss:{:.5f} Acc: {:.3f}% ".format(epoch_loss / val_len, epoch_acc / val_len * 100))
    print('-------------------------------------')
    return epoch_acc / val_len * 100


def load_programs():
    # 加载 data
    input_df = pd.read_csv(train_with_label)
    data_x, data_y = input_df['cut_name'].values, input_df['category3_new'].values
    category_classes = input_df['category3_new'].unique()
    # data pre_processing
    preprocess = Preprocess(data_x, sen_len=6)
    # 设置sen_len
    preprocess.length_distribution(data_x)
    embedding = preprocess.create_tokenizer()
    data_x = preprocess.get_pad_word2idx(data_x)
    data_y = preprocess.get_lab2idx(data_y)

    # 加载model
    lstm_model = LSTMNet(
        embedding,
        embedding_dim=200,
        hidden_dim=128,
        num_classes=len(category_classes),
        num_layers=2,
        dropout=0.5,
        requires_grad=False
    ).to(device)
    # 返回model中的参数的总数目
    total = sum(p.numel() for p in lstm_model.parameters())
    trainable = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    return data_x, data_y, lstm_model


def search_best_dataset(data_x, data_y, model):
    best_x_train, best_y_train = None, None
    best_x_test, best_y_test = None, None
    # 使用k折交叉验证
    kf_10 = KFold(n_splits=10)
    k = 0
    best_accuracy = 0.
    for t_train, t_test in kf_10.split(data_x, data_y):
        print('====================第{}折==================='.format(k))
        k += 1
        train_ds = DefineDataset(data_x[t_train], data_y[t_train])
        train_ip = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
        test_ds = DefineDataset(data_x[t_test], data_y[t_test])
        test_ip = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)
        accuracy_list = list()
        # run epochs
        for ep in range(epochs):
            training(train_ip, model)
            ep_percent = predicting(test_ip, model)
            accuracy_list.append(round(ep_percent, 3))
        mean_accuracy = np.mean(accuracy_list)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_x_train, best_y_train = data_x[t_train], data_y[t_train]
            best_x_test, best_y_test = data_x[t_test], data_y[t_test]
        print('Mean-Accuracy: {:.3f}'.format(mean_accuracy))
    print('Best model with acc {:.3f}%'.format(best_accuracy))
    return best_x_train, best_y_train, best_x_test, best_y_test


if __name__ == '__main__':
    d_x, d_y, classify_model = load_programs()

    # K折交叉验证
    # search_best_dataset(d_x, d_y, classify_model)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(d_x, d_y, test_size=0.3, random_state=5)
    # 构造Dataset
    train_dataset = DefineDataset(x_train, y_train)
    val_dataset = DefineDataset(x_test, y_test)
    # preparing the training loader
    train_input = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print('Training loader prepared.')
    # preparing the validation loader
    val_input = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    print('Validation loader prepared.')

    best_acc = 0.
    # run epochs
    for epoch in range(epochs):
        print('[ Epoch{}: batch_size({}) ]'.format(epoch + 1, batch_size))
        # train for one epoch
        training(train_input, classify_model)
        # predict on validation set
        epoch_percent = predicting(val_input, classify_model)
        if epoch_percent > best_acc:
            # 如果 validation 的结果好于之前所有的结果，就把当下的模型保存
            best_acc = epoch_percent
            # torch.save(model, "{}/ckpt.model".format(model_dir))
            print('saving model with acc {:.3f}%'.format(epoch_percent))
