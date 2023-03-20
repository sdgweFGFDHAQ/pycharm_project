import pandas as pd
from gensim.models import Word2Vec, KeyedVectors

# from utils import eda_class
from workplace.fewsamples.utils import eda_class


class read_file_data:
    def __init__(self):
        self.path_zzx_standard_data = '/home/data/temp/zzx/standard_data/'
        self.segment_number = 12

    def __iter__(self):
        for i in range(self.segment_number):
            path = self.path_zzx_standard_data + 'standard_store_' + str(i) + '.csv'
            df_i = pd.read_csv(path, usecols=['name', 'category3_new', 'cut_name'], keep_default_na=False)
            for cut_name in df_i['cut_name'].values:
                yield cut_name


def get_word2vec():
    # 训练大语料库
    name_iter = read_file_data()
    vec = Word2Vec(sentences=name_iter, vector_size=200, min_count=5, window=2, workers=4, sg=1, epochs=5)
    vec.save('./models/word2vec.model')


def set_word2vec():
    # 增量训练
    few_df = pd.read_csv('./data/few_shot.csv', index_col=0)
    name_list1 = list()
    for name in few_df['cut_name']:
        name_list1.append(name)
    vec = Word2Vec.load('./models/word2vec.model')
    vec.build_vocab(name_list1, update=True)
    vec.train(name_list1, total_examples=vec.corpus_count, epochs=5)
    vec.wv.save_word2vec_format('word2vec.vector')


def data_grow(df):
    print('data_grow', df.head(3))
    # print(len(df.index))
    new_name_list, new_cut_name_list, new_category_list = list(), list(), list()
    vec = KeyedVectors.load_word2vec_format('./models/word2vec.vector')
    eda = eda_class.EDA(num_aug=5, synonyms_model=vec)
    df.apply(random_replace, args=[eda, new_name_list, new_cut_name_list, new_category_list], axis=1)
    new_id_list = ['111110000000000000' + str(id_i) for id_i in range(len(new_name_list))]
    new_df = pd.DataFrame(
        {'id': new_id_list, 'name': new_cut_name_list, 'category3_new': new_category_list, 'cut_name': new_name_list})
    print(new_df.head(3))
    df = pd.concat([df, new_df])
    df.drop_duplicates(subset=['cut_name'], keep='first', inplace=True)
    return df


def random_replace(df, eda_object, name_list, cut_name_list, category_list):
    cn_lists = df['cut_name'].split(' ')
    for_count = 0
    new_name_list = list()
    while (len(new_name_list) < 5) and (for_count < 10):
        # 相似词替换
        # new_cut_name = eda_object.synonym_replacement(cn_lists, n=1)
        # 随机交换
        new_cut_name = eda_object.random_swap(cn_lists, n=1)
        if new_cut_name not in new_name_list:
            new_name_list.append(new_cut_name)
            cut_name_list.append(''.join(new_cut_name))
            category_list.append(df['category3_new'])
        for_count += 1
    name_list.extend(new_name_list)
    # print('增加数量:{}, 循环次数:{}'.format(len(new_name_list), for_count))


if __name__ == '__main__':
    # get_word2vec()
    # set_word2vec()
    w2c_model = Word2Vec.load('word2vec.model')
    w = ['文具', '饭', '便利店', '串串香']
    for i in w:
        word = w2c_model.wv.similar_by_word(i)
        print(word)
