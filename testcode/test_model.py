import random

import numpy as np
import pandas as pd
import torch
from icecream import ic
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
import random
import torch.optim as optim
from sklearn.metrics import f1_score, multilabel_confusion_matrix


def load_model():
    model_name = 'bert-base-chinese'
    MODEL_PATH = 'pytorch_model.bin'
    # a. 通过词典导入分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # b. 导入配置文件
    model_config = BertConfig.from_pretrained(model_name)
    # 修改配置
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    # 通过配置和路径导入模型
    bert_model = BertModel.from_pretrained(MODEL_PATH, config=model_config)
    print(tokenizer.encode('吾儿莫慌'))  # [101, 1434, 1036, 5811, 2707, 102]

    sen_code = tokenizer.encode_plus('这个故事没有终点', "正如星空没有彼岸")
    # print(sen_code)
    # [101, 1434, 1036, 5811, 2707, 102]
    #  {'input_ids': [101, 6821, 702, 3125, 752, 3766, 3300, 5303, 4157, 102, 3633, 1963, 3215, 4958, 3766, 3300, 2516, 2279, 102],
    #  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    print(tokenizer.convert_ids_to_tokens(sen_code['input_ids']))

    # ['[CLS]', '这', '个', '故', '事', '没', '有', '终', '点', '[SEP]', '正', '如', '星', '空', '没', '有', '彼', '岸', '[SEP]']
    # 对编码进行转换，以便输入Tensor
    tokens_tensor = torch.tensor([sen_code['input_ids']])  # 添加batch维度并,转换为tensor,torch.Size([1, 19])
    segments_tensors = torch.tensor(sen_code['token_type_ids'])  # torch.Size([19])

    bert_model.eval()

    # 进行编码
    with torch.no_grad():
        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
        encoded_layers = outputs  # outputs类型为tuple

        print(encoded_layers[0].shape, encoded_layers[1].shape,
              encoded_layers[2][0].shape, encoded_layers[3][0].shape)
        # torch.Size([1, 19, 768]) torch.Size([1, 768])
        # torch.Size([1, 19, 768]) torch.Size([1, 12, 19, 19])
    model_name = 'bert-base-chinese'  # 指定需下载的预训练模型参数

    # 任务一：遮蔽语言模型
    # BERT 在预训练中引入 [CLS] 和 [SEP] 标记句子的 开头和结尾
    samples = ['[CLS] 中国的首都是哪里？ [SEP] 北京是 [MASK] 国的首都。 [SEP]']  # 准备输入模型的语句

    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer_text = [tokenizer.tokenize(i) for i in samples]  # 将句子分割成一个个token，即一个个汉字和分隔符
    # [['[CLS]', '中', '国', '的', '首', '都', '是', '哪', '里', '？', '[SEP]', '北', '京', '是', '[MASK]', '国', '的', '首', '都', '。', '[SEP]']]
    # print(tokenizer_text)

    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenizer_text]
    input_ids = torch.LongTensor(input_ids)
    # print(input_ids)
    # tensor([[ 101,  704, 1744, 4638, 7674, 6963, 3221, 1525, 7027, 8043,  102, 1266,
    #           776, 3221,  103, 1744, 4638, 7674, 6963,  511,  102]])

    # 读取预训练模型
    model = BertForMaskedLM.from_pretrained(model_name, cache_dir='F:/Transformer-Bert/')
    model.eval()

    outputs = model(input_ids)
    prediction_scores = outputs[0]  # prediction_scores.shape=torch.Size([1, 21, 21128])
    sample = prediction_scores[0].detach().numpy()  # (21, 21128)

    # 21为序列长度，pred代表每个位置最大概率的字符索引
    pred = np.argmax(sample, axis=1)  # (21,)
    # ['，', '中', '国', '的', '首', '都', '是', '哪', '里', '？', '。', '北', '京', '是', '中', '国', '的', '首', '都', '。', '。']
    print(tokenizer.convert_ids_to_tokens(pred))
    print(tokenizer.convert_ids_to_tokens(pred)[14])  # 被标记的[MASK]是第14个位置, 中
    # sen_code1 = tokenizer.encode_plus('今天天气怎么样？', '今天天气很好！')
    # sen_code2 = tokenizer.encode_plus('明明是我先来的！', '我喜欢吃西瓜！')

    # tokens_tensor = torch.tensor([sen_code1['input_ids'], sen_code2['input_ids']])
    # print(tokens_tensor)
    # tensor([[ 101,  791, 1921, 1921, 3698, 2582,  720, 3416,  102,  791, 1921, 1921,
    #          3698, 2523, 1962,  102],
    #         [ 101, 3209, 3209, 3221, 2769, 1044, 3341, 4638,  102, 7471, 3449,  679,
    #          1963, 1921, 7360,  102]])

    # 上面可以换成
    samples = ["[CLS]天气真的好啊！[SEP]一起出去玩吧！[SEP]", "[CLS]小明今年几岁了[SEP]我不喜欢学习！[SEP]"]
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenized_text = [tokenizer.tokenize(i) for i in samples]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    tokens_tensor = torch.LongTensor(input_ids)


def test_KNsamples():
    # 创建DataFrame
    df = pd.DataFrame({
        'category': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B',
                     'C', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D',
                     'E', 'E', 'E', 'E', 'E'],
        'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25]
    })


def createData():
    text_list_pos = ["电影内容很好", "电影题材很好", "演员演技很好", "故事很感人", "电影特效很好"]
    text_list_neg = ["电影内容垃圾", "电影是真的垃圾", "表演太僵硬了", "故事又臭又长", "电影太让人失望了"]
    test_pos = ["电影", "很", "好"]
    test_neg = ["电影", "垃圾"]
    words_pos = [[item for item in jieba.cut(text)] for text in text_list_pos]
    words_neg = [[item for item in jieba.cut(text)] for text in text_list_neg]
    words_all = []
    for item in words_pos:
        for key in item:
            words_all.append(key)
    for item in words_neg:
        for key in item:
            words_all.append(key)
    vocab = list(set(words_all))
    word2idx = {w: c for c, w in enumerate(vocab)}
    idx_words_pos = [[word2idx[item] for item in text] for text in words_pos]
    idx_words_neg = [[word2idx[item] for item in text] for text in words_neg]
    idx_test_pos = [word2idx[item] for item in test_pos]
    idx_test_neg = [word2idx[item] for item in test_neg]
    return vocab, word2idx, idx_words_pos, idx_words_neg, idx_test_pos, idx_test_neg


def createOneHot(vocab, idx_words_pos, idx_words_neg, idx_test_pos, idx_test_neg):
    input_dim = len(vocab)
    features_pos = torch.zeros(size=[len(idx_words_pos), input_dim])
    features_neg = torch.zeros(size=[len(idx_words_neg), input_dim])
    for i in range(len(idx_words_pos)):
        for j in idx_words_pos[i]:
            features_pos[i, j] = 1.0

    for i in range(len(idx_words_neg)):
        for j in idx_words_neg[i]:
            features_neg[i, j] = 1.0
    features = torch.cat([features_pos, features_neg], dim=0)
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    labels = torch.LongTensor(labels)
    test_x_pos = torch.zeros(size=[1, input_dim])
    test_x_neg = torch.zeros(size=[1, input_dim])
    for item in idx_test_pos:
        test_x_pos[0, item] = 1.0
    for item in idx_test_neg:
        test_x_neg[0, item] = 1.0
    test_x = torch.cat([test_x_pos, test_x_neg], dim=0)
    test_labels = torch.LongTensor([1, 0])
    return features, labels, test_x, test_labels


def randomGenerate(features):
    N = features.shape[0]
    half_n = N // 2
    support_input = torch.zeros(size=[6, features.shape[1]])
    query_input = torch.zeros(size=[4, features.shape[1]])
    postive_list = list(range(0, half_n))
    negtive_list = list(range(half_n, N))
    support_list_pos = random.sample(postive_list, 3)
    support_list_neg = random.sample(negtive_list, 3)
    query_list_pos = [item for item in postive_list if item not in support_list_pos]
    query_list_neg = [item for item in negtive_list if item not in support_list_neg]
    index = 0
    for item in support_list_pos:
        support_input[index, :] = features[item, :]
        index += 1
    for item in support_list_neg:
        support_input[index, :] = features[item, :]
        index += 1
    index = 0
    for item in query_list_pos:
        query_input[index, :] = features[item, :]
        index += 1
    for item in query_list_neg:
        query_input[index, :] = features[item, :]
        index += 1
    query_label = torch.LongTensor([1, 1, 0, 0])
    return support_input, query_input, query_label


class fewModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(fewModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        # 线性层进行编码
        self.linear = nn.Linear(input_dim, hidden_dim)

    def embedding(self, features):
        result = self.linear(features)
        return result

    def forward(self, support_input, query_input):

        support_embedding = self.embedding(support_input)
        query_embedding = self.embedding(query_input)
        support_size = support_embedding.shape[0]
        every_class_num = support_size // self.num_class
        class_meta_dict = {}
        for i in range(0, self.num_class):
            class_meta_dict[i] = torch.sum(support_embedding[i * every_class_num:(i + 1) * every_class_num, :],
                                           dim=0) / every_class_num
        class_meta_information = torch.zeros(size=[len(class_meta_dict), support_embedding.shape[1]])
        for key, item in class_meta_dict.items():
            class_meta_information[key, :] = class_meta_dict[key]
        N_query = query_embedding.shape[0]
        result = torch.zeros(size=[N_query, self.num_class])
        for i in range(0, N_query):
            temp_value = query_embedding[i].repeat(self.num_class, 1)
            cosine_value = torch.cosine_similarity(class_meta_information, temp_value, dim=1)
            result[i] = cosine_value
        result = F.log_softmax(result, dim=1)
        return result


hidden_dim = 4
n_class = 2
lr = 0.01
epochs = 1000
vocab, word2idx, idx_words_pos, idx_words_neg, idx_test_pos, idx_test_neg = createData()
features, labels, test_x, test_labels = createOneHot(vocab, idx_words_pos, idx_words_neg, idx_test_pos, idx_test_neg)

model = fewModel(features.shape[1], hidden_dim, n_class)
optimer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)


def train(epoch, support_input, query_input, query_label):
    optimer.zero_grad()
    output = model(support_input, query_input)
    loss = F.nll_loss(output, query_label)
    loss.backward()
    optimer.step()
    print("Epoch: {:04d}".format(epoch), "loss:{:.4f}".format(loss))


def accu():
    # 假设有真实标签和预测标签
    y_true = [[1, 0, 1, 0],
              [0, 1, 1, 0],
              [1, 1, 0, 1]]

    y_pred = [[1, 0, 0, 1],
              [1, 1, 1, 0],
              [0, 1, 0, 1]]

    # 计算每个标签的 F1 分数
    f1_scores = f1_score(y_true, y_pred, average='macro')
    print("F1 Scores for each label:", f1_scores)

    # 计算平均 F1 分数
    average_f1_score = f1_score(y_true, y_pred, average='weighted')
    print("Average F1 Score:", average_f1_score)

    sds = multilabel_confusion_matrix(y_true, y_pred)
    print(sds)


if __name__ == '__main__':
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    print(df1)
    odict = {'a1': df1['a']}
    odict['b1'] = df1['b'].values
    odict['c1'] = [7, 8, 9]
    df2 = pd.DataFrame(odict)
    print(df2)
