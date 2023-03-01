# encoding=utf-8
from ast import literal_eval
import jieba
import pandas as pd
import numpy as np
from collections import Counter

from gensim.models import Word2Vec
from imblearn.over_sampling import BorderlineSMOTE
from icecream import ic
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from workplace.fewsamples.w2c_eda import cut_word, data_grow
from workplace.fewsamples.global_parameter import StaticParameter as SP


def get_few_shot():
    csv = pd.read_csv('../standard_store_gz.csv', usecols=['id', 'name', 'category3_new', 'cut_name'])
    sample = csv.sample(n=2000, ignore_index=True, random_state=11)
    # 设置不可分割的词
    jieba.load_userdict('indiv_words.txt')
    sample['cut_name'] = sample['name'].apply(cut_word)
    sample.to_csv('few_shot.csv')
    return sample


def get_few_data(grow_df):
    few_df = pd.read_csv('few_shot.csv', index_col=0)
    category_list = few_df[['category3_new']].drop_duplicates(keep='first')
    category_num = len(category_list.index)
    category_list = category_list['category3_new'].values
    print('category_num:%s' % category_num, 'category_list:%s' % category_list)
    # 扩展少于k_neighbors数的类别
    input_df = few_df.copy()
    low_neighbors_df = input_df.groupby('category3_new').filter(lambda cn: len(cn) < 6)
    new_data_df = pd.DataFrame(columns=['id', 'name', 'category3_new', 'cut_name_new'])
    if not grow_df.empty:
        new_data_df = pd.concat([low_neighbors_df, grow_df], ignore_index=True)
        print("扩展后数据量：", len(few_df.index))
    new_data_df.to_csv('input_data.csv')
    # 文本向量化
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(new_data_df['cut_name_new'].values)
    x = tokenizer.texts_to_sequences(new_data_df['cut_name_new'].values)
    x = pad_sequences(x, maxlen=SP.MAX_LENGTH)
    y = pd.get_dummies(new_data_df['category3_new']).values
    # smote数据增强
    bl_smote = BorderlineSMOTE(k_neighbors=5, kind='borderline-1')  # ADASYN\SVMSMOTE\KMeansSMOTE
    x_bls, y_bls = bl_smote.fit_resample(x, y)
    return x_bls, y_bls


def model_train(x, y):
    # 拆分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.10, random_state=42)
    # 定义模型
    model = Sequential()
    model.add(Embedding(SP.MAX_WORDS_NUM, SP.EMBEDDING_DIM, input_length=x.shape[1]))
    model.add(SpatialDropout1D(rate=0.2, seed=12))
    model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, ))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    print('=================================', history.history)
    accuracy = model.evaluate(X_test, Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accuracy[0], accuracy[1]))
    return x, y


if __name__ == '__main__':
    # get_few_shot()
    w2c_model = Word2Vec.load('word2vec.model')
    grow_df = data_grow(w2c_model)
    x_d, y_d = get_few_data(grow_df)
    model_train(x_d, y_d)
