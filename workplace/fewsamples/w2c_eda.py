import jieba
import pandas as pd
import re
from ast import literal_eval
from gensim.models import Word2Vec
from workplace.fewsamples import eda_class


def get_word2vec():
    # 训练大语料库
    few_df = pd.read_csv('../all_labeled_data.csv', index_col=0)
    print(few_df.head(2))
    name_list = list()
    few_df['cut_name'] = few_df['cut_name'].apply(literal_eval)
    for name in few_df['cut_name']:
        name_list.append(name)
    vec = Word2Vec(sentences=name_list, vector_size=6, min_count=1, window=2, workers=4)
    vec.save('word2vec.model')


def set_word2vec():
    # 增量训练
    few_df = pd.read_csv('few_shot.csv', index_col=0)
    name_list1 = list()
    few_df['cut_name'] = few_df['cut_name'].apply(literal_eval)
    for name in few_df['cut_name']:
        name_list1.append(name)
    vec = Word2Vec.load('word2vec.model')
    vec.build_vocab(name_list1, update=True)
    vec.train(name_list1, total_examples=vec.corpus_count, epochs=5)
    vec.wv.save_word2vec_format('word2vec.vector')
    return vec


def data_grow(vec, df):
    print('data_grow', df.head(3))
    # print(len(df.index))
    new_name_list, new_cut_name_list, new_category_list = list(), list(), list()
    eda = eda_class.EDA(num_aug=5, synonyms_model=vec)
    df.apply(random_replace, args=[eda, new_name_list, new_cut_name_list, new_category_list], axis=1)
    new_id_list = ['990000000000000000' + str(i) for i in range(len(new_name_list))]
    new_df = pd.DataFrame({'id': new_id_list, 'name': new_cut_name_list, 'category3_new': new_category_list, 'cut_name': new_name_list})
    print(new_df.head(3))
    df = pd.concat([df, new_df])
    df.drop_duplicates(subset=['cut_name'], keep='first', inplace=True)
    print(df.head(3))
    return df


def random_replace(df, eda_object, name_list, cut_name_list, category_list):
    cn_lists = literal_eval(df['cut_name'])
    for_count = 0
    new_name_list = list()
    while (len(new_name_list) < 5) and (for_count < 20):
        new_cut_name = eda_object.synonym_replacement(cn_lists, n=1)
        if new_cut_name not in new_name_list:
            new_name_list.append(new_cut_name)
            cut_name_list.append(''.join(new_cut_name))
            category_list.append(df['category3_new'])
        for_count += 1
    name_list.extend(new_name_list)
    # print('增加数量:{}, 循环次数:{}'.format(len(new_name_list), for_count))


def cut_word(word):
    out_word_list = []
    # 清洗特殊字符
    word = re.sub(r'\(.*?\)|[^a-zA-Z0-9\u4e00-\u9fa5]|(丨)', ' ', str(word))
    # 形如:"EXO店x铺excelAxB" 去除x
    word = re.sub(r'(?<=[\u4e00-\u9fa5])([xX])(?=[\u4e00-\u9fa5])|(?<=[A-Z])x(?=[A-Z])', ' ', word)
    l_cut_words = jieba.lcut(word)
    # 人工去除明显无用的词
    stop_words = [line.strip() for line in open('../stopwords.txt', 'r', encoding='utf-8').readlines()]
    for lc_word in l_cut_words:
        if lc_word not in stop_words:
            if lc_word != '\t' and not lc_word.isspace():
                out_word_list.append(lc_word)
    return out_word_list


if __name__ == '__main__':
    # get_word2vec()
    # set_word2vec()
    w2c_model = Word2Vec.load('word2vec.model')
    w = ['陈体', '猪脚', '龙虾', '鲜肉', '豆腐花']
    for i in w:
        word = w2c_model.wv.similar_by_word(i)
        print(word)
