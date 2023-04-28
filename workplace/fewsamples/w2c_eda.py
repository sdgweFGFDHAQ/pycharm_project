import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from icecream import ic

from utils import eda_class


# from workplace.fewsamples.utils import eda_class


class read_file_data:
    def __init__(self):
        self.path_zzx_standard_data = '/home/data/temp/zzx/standard_data/'
        self.segment_number = 12
        self.columns = ['category3_new', 'cut_name']

    def __iter__(self):
        no_standard_value_count = 0
        for i in range(self.segment_number):
            path = self.path_zzx_standard_data + 'standard_store_' + str(i) + '.csv'
            df_i = pd.read_csv(path, usecols=self.columns)
            for cut_name in df_i['cut_name'].values:
                if type(cut_name) is str:
                    yield cut_name.split()
                elif type(cut_name) is list:
                    yield ' '.join(cut_name).split()
                else:
                    no_standard_value_count += 1
        print("columns type error: should be str or list! error_count:{}".format(no_standard_value_count))


def get_word2vec():
    # 训练大语料库
    name_iter = read_file_data()
    vec = Word2Vec(sentences=name_iter, vector_size=200, min_count=3, window=2, workers=4, sg=1, epochs=5)
    vec.save('./models/word2vec.model')


def set_word2vec(column):
    """
    :param column: str
    :return:
    """
    # 增量训练
    few_df = pd.read_csv('./data/few_shot.csv', index_col=0)
    name_list = list()
    if few_df[column].dtype is str:
        for name in few_df[column].values:
            name_list.append(name.split())
    vec = Word2Vec.load('./models/word2vec.model')
    vec.build_vocab(name_list, update=True)
    vec.train(name_list, total_examples=vec.corpus_count, epochs=5)
    vec.wv.save_word2vec_format('./models/word2vec.vector')


def data_grow(df, column_list):
    ic('data_grow', df.head(2))
    # print(len(df.index))
    columns_dict = dict()
    for col in column_list:
        columns_dict[col] = list()
    vec = KeyedVectors.load_word2vec_format('./models/word2vec.vector')
    eda = eda_class.EDA(num_aug=5, synonyms_model=vec)
    df.apply(random_replace, args=[eda, columns_dict], axis=1)
    new_df = pd.DataFrame(columns_dict)
    ic(new_df.head(3))
    df = pd.concat([df, new_df])
    df.drop_duplicates(subset=['cut_name'], keep='first', inplace=True)
    return df


def random_replace(df, eda_object, col_dict):
    cn_list = df['cut_name'].split(' ')
    # 相似词替换
    syn_num = random_replace_syn(cn_list, eda_object, col_dict['name'], col_dict['cut_name'])
    # 随机交换
    swap_num = random_replace_swap(cn_list, eda_object, col_dict['name'], col_dict['cut_name'])
    for i in range(syn_num + swap_num):
        for col_k, v in col_dict.items():
            if col_k == 'store_id':
                col_dict[col_k].append(str(df[col_k]) + str(i))
            elif col_k == 'name' or col_k == 'cut_name':
                continue
            else:
                col_dict[col_k].append(df[col_k])


def random_replace_syn(cn_list, eda_object, name_list, cut_name_list):
    # min_num最少生成数据条数;max_num最大操作次数
    min_num, max_num = 0, 0
    new_name_list = list()
    while (min_num < 8) and (max_num < 10):
        # 相似词替换
        new_cut_name = eda_object.synonym_replacement(cn_list, n=1)
        name_str = ''.join(new_cut_name)
        cut_name_str = ' '.join(new_cut_name)
        if new_cut_name not in new_name_list:
            name_list.append(name_str)
            cut_name_list.append(cut_name_str)
            new_name_list.append(name_str)
            min_num += 1
        max_num += 1
    # print('增加数量:{}, 循环次数:{}'.format(len(new_name_list), for_count))
    return len(new_name_list)


def random_replace_swap(cn_list, eda_object, name_list, cut_name_list):
    # min_num最少生成数据条数;max_num最大操作次数
    min_num, max_num = 0, 0
    new_name_list = list()
    while (min_num < 6) and (max_num < 10):
        # 随机交换
        new_cut_name = eda_object.random_swap(cn_list, n=2)
        name_str = ''.join(new_cut_name)
        cut_name_str = ' '.join(new_cut_name)
        if new_cut_name not in new_name_list:
            name_list.append(name_str)
            cut_name_list.append(cut_name_str)
            new_name_list.append(name_str)
            min_num += 1
        max_num += 1
    # print('增加数量:{}, 循环次数:{}'.format(len(new_name_list), for_count))
    return len(new_name_list)


if __name__ == '__main__':
    get_word2vec()
    set_word2vec('cut_name')
    w2c_model = Word2Vec.load('./models/word2vec.model')
    w = ['文具', '饭', '便利店', '串串香']
    for i in w:
        word = w2c_model.wv.similar_by_word(i)
        print(word)
