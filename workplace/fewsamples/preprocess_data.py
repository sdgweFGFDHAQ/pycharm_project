# encoding=utf-8

from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import torch

from workplace.fewsamples.w2c_eda import data_grow
from workplace.fewsamples.utils.mini_tool import set_jieba, cut_word

# 原始文件路径
original_file_path = '../all_labeled_data.csv'


# 读取原始文件,将数据格式标准化
def set_file_standard_data(path, is_label=True):
    """
    :param path:
    :param is_label: 是否清洗无标签数据,默认为 True
    """
    csv_data = pd.read_csv(path,
                           usecols=['id', 'name', 'category1_new', 'category2_new', 'category3_new'],
                           keep_default_na=False)
    if is_label:
        csv_data = csv_data[csv_data['category1_new'].notnull() & (csv_data['category1_new'] != "")]
        # 用一级标签填充空白(NAN)的二级标签、三级标签
        csv_data['category2_new'].fillna(csv_data['category1_new'], inplace=True)
        csv_data['category3_new'].fillna(csv_data['category2_new'], inplace=True)
        # 得到各级标签映射字典
        category = csv_data[['category1_new', 'category2_new', 'category3_new']]
        category = category.drop_duplicates(keep='first')
        category.reset_index(inplace=True, drop=True)
        category.to_csv('./data/category_dict.csv')
        print("类别个数：", len(category['category3_new']))
        # 得到标准数据
        set_jieba()
        csv_data['cut_name'] = csv_data['name'].apply(cut_word)
        csv_data.to_csv('./data/labeled_data.csv', columns=['id', 'name', 'category3_new', 'cut_name'])
    else:
        csv_data = csv_data[csv_data['category1_new'].null() | (csv_data['category1_new'] == "")]
        # 得到标准数据
        set_jieba()
        csv_data['cut_name'] = csv_data['name'].apply(cut_word)
        csv_data.to_csv('./data/unlabeled_data.csv', columns=['id', 'name', 'category3_new', 'cut_name'])


class Preprocess:
    # 标准化的已打标数据集
    labeled_data_path = './data/labeled_data.csv'
    # 标准化的未打标数据集
    unlabeled_data_path = './data/unlabeled_data.csv'
    # 取样少样本数据集
    few_shot_path = './data/few_shot.csv'

    def __init__(self, sentences, sen_len):  # 首先定义类的一些属性
        self.embedding = KeyedVectors.load_word2vec_format('./models/word2vec.vector')
        self.vector_size = self.embedding.vector_size
        self.sentences = sentences
        self.sen_len = sen_len
        self.word2idx = {}
        self.idx2word = []
        self.embedding_matrix = []
        self.lab2idx = {}
        self.idx2lab = {}

    # 读取全量数据
    def get_data(self, is_label=True):
        """
        :param is_label: 是否读取有标签数据,默认为 True
        :return:
        """
        if is_label:
            path = self.labeled_data_path
            csv_data = pd.read_csv(path, usecols=['id', 'name', 'category3_new', 'cut_name'])
            return csv_data
        else:
            path = self.unlabeled_data_path
            csv_data = pd.read_csv(path, usecols=['id', 'name', 'category3_new', 'cut_name'])
            return csv_data

    # 从原始数据获取小样本，统计类别，生成”类别-》id索引“的字典
    def get_few_shot(self):
        sample = self.get_data()
        few_df = sample.groupby(sample['category3_new']).sample(n=50, random_state=11, replace=True).drop_duplicates(
            keep='first')
        few_df = few_df.sample(frac=1)
        few_df.to_csv(self.few_shot_path)
        category_list = few_df[['category3_new']].drop_duplicates(keep='first')
        category_num = len(category_list.index)
        print('数据类别个数:', category_num)
        # 生成类别-id字典
        few_df['cat_id'] = few_df['category3_new'].factorize()[0]
        cat_df = few_df[['category3_new', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(
            drop=True)
        cat_df.to_csv('./data/category_to_id.csv')
        # 生成类别字典
        self.lab2idx = dict(zip(cat_df['category3_new'], cat_df['cat_id']))
        self.idx2lab = dict(zip(cat_df['cat_id'], cat_df['category3_new']))

    # 数据增强
    def grow_few_data(self):
        # 扩展少于k_neighbors数的类别
        old_df = pd.read_csv(self.few_shot_path, index_col=0)
        new_data_df = data_grow(old_df)
        new_data_df = new_data_df.sample(frac=1).reset_index()
        print("扩展后数据量：", len(new_data_df.index))
        new_data_df.to_csv('./data/input_data.csv')

    # 构建向量矩阵
    def create_tokenizer(self):
        # 把'<PAD>'、'<UNK>'加进embedding
        vocab_list = list()
        vocab_list.append('<PAD>')
        vocab_list = vocab_list + [k for k in self.embedding.key_to_index.keys()]
        vocab_list.append('<UNK>')
        embeddings_matrix = np.zeros((len(vocab_list) + 2, self.vector_size))
        for i in range(len(vocab_list)):
            try:
                word = vocab_list[i]
                embeddings_matrix[i] = self.embedding[word]
            except Exception:
                vector = torch.empty(1, self.vector_size)
                torch.nn.init.uniform_(vector)
                embeddings_matrix[i] = vector
            finally:
                self.word2idx[vocab_list[i]] = i
        self.idx2word = vocab_list
        self.embedding_matrix = torch.tensor(embeddings_matrix)
        return self.embedding_matrix

    # 把句子里面的字变成相对应的index
    def get_pad_word2idx(self, sentences):
        text_to_sequence = []
        for sentence in sentences:
            sequence = []
            for word in sentence:
                if word in self.word2idx.keys():
                    sequence.append(self.word2idx[word])
                else:
                    sequence.append(self.word2idx['<UNK>'])
            # 统一句子长度
            sequence = self.pad_sequence(sequence)
            text_to_sequence.append(sequence)
        return torch.tensor(text_to_sequence)

    # 填充长度,将每个句子变成一样的长度
    def pad_sequence(self, sentence):
        if len(sentence) > self.sen_len:
            new_sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            new_sentence = list()
            for _ in range(pad_len):
                new_sentence.append(self.word2idx["<PAD>"])
            new_sentence.extend(sentence)
        assert len(new_sentence) == self.sen_len
        return new_sentence

    # 获取数据“标签列“的向量形式
    def get_lab2idx(self, labels):
        # 输出1D标签索引
        y = list()
        for lab in labels:
            y.append(self.lab2idx[lab])
        print('lab2idx:', y)
        return torch.LongTensor(y)

    # 店名长度的分布分析
    def length_distribution(self, sentences):
        len_list = []
        for sentence in sentences:
            len_list.append(len(sentence))
        self.sen_len = int(np.median(np.array(pd.value_counts(len_list).keys())))
        print('length_distribution:', self.sen_len)


if __name__ == '__main__':
    set_file_standard_data(original_file_path)
