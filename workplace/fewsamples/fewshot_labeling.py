# encoding=utf-8
from ast import literal_eval
import jieba
import pandas as pd
import numpy as np
from collections import Counter

from gensim.models import Word2Vec
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
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
    sample = pd.read_csv('../all_labeled_data.csv', usecols=['id', 'name', 'category3_new', 'cut_name']) \
        .sample(n=2000, ignore_index=True, random_state=11)
    # 设置不可分割的词
    # jieba.load_userdict('indiv_words.txt')
    # sample['cut_name'] = sample['name'].apply(cut_word)
    sample.to_csv('few_shot.csv')
    category_list = sample[['category3_new']].drop_duplicates(keep='first')
    category_num = len(category_list.index)
    print('原始数据类别个数:', category_num)


def get_few_data():
    # 扩展少于k_neighbors数的类别
    old_df = pd.read_csv('few_shot.csv', index_col=0)
    w2c_model = Word2Vec.load('word2vec.model')
    new_data_df = data_grow(w2c_model, old_df)
    print("扩展后数据量：", len(new_data_df.index))
    # 生成类别-id字典
    new_data_df['cat_id'] = new_data_df['category3_new'].factorize()[0]
    cat_df = new_data_df[['category3_new', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
    cat_df.to_csv('category_to_id.csv')
    new_data_df.to_csv('input_data.csv')


def model_train():
    new_data_df = pd.read_csv('input_data.csv')
    # 文本向量化
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(new_data_df['cut_name'].values)
    print('不同词语个数：', len(tokenizer.word_index))
    x = tokenizer.texts_to_sequences(new_data_df['cut_name'].values)
    x = pad_sequences(x, maxlen=SP.MAX_LENGTH)
    y = pd.get_dummies(new_data_df['category3_new']).values
    # smote数据增强
    bl_smote = BorderlineSMOTE(k_neighbors=5, kind='borderline-1')  # ADASYN\SVMSMOTE\KMeansSMOTE
    X, Y = bl_smote.fit_resample(x, y)
    # 拆分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    # 定义模型
    model = Sequential()
    model.add(Embedding(SP.MAX_WORDS_NUM, SP.EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(rate=0.2, seed=12))
    model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, ))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # history = model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.1,
    #                     callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    # print('================-------------===============', history.history)
    # accuracy = model.evaluate(X_test, Y_test)
    # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accuracy[0], accuracy[1]))
    model.fit(X, Y, epochs=5, batch_size=64, validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    return tokenizer, model


def predict_result(tokenizer, model):
    df = pd.read_csv('../all_labeled_data.csv').sample(n=30000, random_state=22)
    test_lists = list()
    for i in df['cut_name']:
        i = literal_eval(i)
        test_lists.append(' '.join(i))
    seq = tokenizer.texts_to_sequences(test_lists)
    padded = pad_sequences(seq, maxlen=SP.MAX_LENGTH)
    pred_lists = model.predict(padded)
    # accuracy = model.evaluate(padded, pd.get_dummies(df['category3_new']).values)
    # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accuracy[0], accuracy[1]))
    print(pred_lists[:10])
    id_lists = pred_lists.argmax(axis=1)
    cat_id = pd.read_csv('category_to_id.csv')
    ic_dict = dict(zip(cat_id['cat_id'], cat_id['category3_new']))
    cat_lists = list()
    for id in id_lists:
        cat_lists.append(ic_dict[id])
    result = pd.DataFrame(
        {'store_id': df['id'], 'name': df['name'], 'category3_new': df['category3_new'],
         'predict_category': cat_lists})
    result.to_csv('test_predict_category.csv')


if __name__ == '__main__':
    # 获取少样本数据集
    # get_few_shot()
    # get_few_data()
    # 训练模型预测
    token, mod = model_train()
    predict_result(token, mod)
