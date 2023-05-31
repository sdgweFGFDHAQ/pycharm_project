import numpy
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
PU-Learning，即P代表的是Positive，U代表的是Unlabel，负样本实际上是泛样本
"""
noEDA_prefix = '/home/data/temp/lxb/alchemy/data/noEDA_data'
input_prefix = '/home/data/temp/lxb/alchemy/data/input_dataset'
labeled_path = '../sv_report_data.csv'

batch_size = 32


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


# (1)使用正样本和泛洋本训练分类器
def training(input_loader, query_label, model):
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)
    model.train()
    epoch_los, epoch_acc = 0.0, 0.0
    for i, (support_input, query_input) in enumerate(input_loader):
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
        accu = score(output, query_label)
        epoch_acc += accu.item()
        # 6. 反向传播
        loss.backward()
        # 7. 更新梯度
        optimizer.step()
    loss_value = epoch_los / len(support_loader)
    acc_value = epoch_acc / len(support_loader)
    print("accuracy: {:.2%},loss:{:.4f}".format(acc_value, loss_value))
    return acc_value, loss_value


# (2)对泛样本打分，选取概率最高的作为负样本，重新生成样本集csv文件
def score(pos_path, pred_path):
    # 构建预测集
    positive_df['label'] = 1
    p_samples = positive_df.merge(positive_df, how='left', on='label')
    print(p_samples.head())
    # 模型预测
    model = torch.load('bert_attention.model')
    predict(model, pred_dataloader, pos_path=positive_path, pred_path=predict_path)


def rerun(epoch):
    features = ['name', 'storeType']
    labels = ['碳酸饮料', '果汁', '茶饮', '水', '乳制品', '植物蛋白饮料', '功能饮料']
    columns = ['store_id']
    columns.extend(features)
    columns.extend(labels)

    labeled_df = pd.read_csv(labeled_path, usecols=columns)
    acc_List = []
    init_model = True
    tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese')
    for _ in range(epoch):
        train_loader = get_labeled_dataloader(noEDA_prefix, input_prefix)
        training(train_loader, input_prefix)
        acc = train_by_all_dataset(source_path=input_prefix, is_init=init_model)
        acc_List.append(acc)

        init_model = False
    # (4)最终我们拿到了N次的预测结果，取平均作为最终的预测概率
    mean_acc = numpy.mean(acc_List)
    print(mean_acc)


if __name__ == '__main__':
    rerun(5)
