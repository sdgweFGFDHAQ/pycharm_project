import os

from ast import literal_eval

from icecream import ic
import pandas as pd
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
train_with_label = os.path.join(path_prefix, 'data/input_data.csv')
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


def training(train_loader, model, criterion, optimizer):
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


def predicting(val_loader, model, criterion):
    model.eval()  # 將 model 的模式设定为 eval，固定model的参数
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
    print('-----------------------------------------------')
    return epoch_acc / val_len * 100


if __name__ == '__main__':
    # load data
    input_df = pd.read_csv(train_with_label)
    input_df['cut_name'] = input_df['cut_name'].apply(literal_eval)
    data_x, data_y = input_df['cut_name'].values, input_df['category3_new'].values
    category_classes = input_df['category3_new'].unique()
    # data pre_processing
    preprocess = Preprocess(data_x, sen_len=6)
    # 设置sen_len
    preprocess.length_distribution(data_x)
    embedding = preprocess.create_tokenizer()
    data_x = preprocess.sentence_word2idx(data_x)
    data_y = preprocess.labels_to_tensor(data_y)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=5)
    # 构造Dataset
    train_dataset = DefineDataset(x_train, y_train)
    val_dataset = DefineDataset(x_train, y_train)
    # preparing the training loader
    train_input = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print('Training loader prepared.')
    # preparing the validation loader
    val_input = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    print('Validation loader prepared.')

    # load model
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

    # 多分类损失函数
    crit = nn.CrossEntropyLoss()
    # crit = nn.CrossEntropyLoss(reduction='sum')
    # 使用Adam优化器
    optim_adam = optim.Adam(lstm_model.parameters(), lr=lr)
    best_acc = 0.
    # run epochs
    for epoch in range(epochs):
        print('[ Epoch{}: batch_size({}) ]'.format(epoch + 1, batch_size))
        # train for one epoch
        training(train_input, lstm_model, crit, optim_adam)
        # predict on validation set
        epoch_percent = predicting(val_input, lstm_model, crit)
        if epoch_percent > best_acc:
            # 如果 validation 的结果好于之前所有的结果，就把当下的模型保存
            best_acc = epoch_percent
            # torch.save(model, "{}/ckpt.model".format(model_dir))
            print('saving model with acc {:.3f}%'.format(epoch_percent))
