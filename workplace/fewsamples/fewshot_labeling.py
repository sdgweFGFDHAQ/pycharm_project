# encoding=utf-8

from ast import literal_eval
from collections import Counter
from gensim.models import Word2Vec, KeyedVectors
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from icecream import ic
import jieba
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from workplace.fewsamples.w2c_eda import cut_word, data_grow
from workplace.fewsamples.global_parameter import StaticParameter as SP


def get_few_shot():
    sample = pd.read_csv('../all_labeled_data.csv', usecols=['id', 'name', 'category3_new', 'cut_name'])
    few_df = sample.groupby(sample['category3_new']).sample(n=50, random_state=11, replace=True).drop_duplicates(
        keep='first')
    few_df = few_df.sample(frac=1)
    # 设置不可分割的词
    # jieba.load_userdict('indiv_words.txt')
    # sample['cut_name'] = sample['name'].apply(cut_word)
    few_df.to_csv('few_shot.csv')
    category_list = few_df[['category3_new']].drop_duplicates(keep='first')
    category_num = len(category_list.index)
    print('原始数据类别个数:', category_num)


def get_few_data():
    # 扩展少于k_neighbors数的类别
    old_df = pd.read_csv('few_shot.csv', index_col=0)
    new_data_df = data_grow(old_df)
    new_data_df = new_data_df.sample(frac=1).reset_index()
    print("扩展后数据量：", len(new_data_df.index))
    # 生成类别-id字典
    new_data_df['cat_id'] = new_data_df['category3_new'].factorize()[0]
    cat_df = new_data_df[['category3_new', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
    cat_df.to_csv('category_to_id.csv')
    new_data_df.to_csv('input_data.csv')


def create_tokenizer(cut_name_list, word_index):
    data = []
    for cut_name in cut_name_list:
        new_txt = []
        for word in cut_name:
            try:
                new_txt.append(word_index[word])  # 把句子中的 词语转化为index
            except Exception:
                new_txt.append(0)
        data.append(new_txt)
    x_seq = pad_sequences(data, maxlen=SP.MAX_LENGTH)  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    return x_seq


def pre_matrix():
    embedding_model = KeyedVectors.load_word2vec_format('word2vec.vector')
    word2idx = {'PAD': 0}
    vocab_list = [k for k in embedding_model.key_to_index.keys()]
    embeddings_matrix = np.zeros((len(vocab_list) + 1, embedding_model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = embedding_model.get_vector(vocab_list[i])
    return word2idx, embeddings_matrix


def model_train():
    # new_data_df = pd.read_csv('few_shot.csv')
    new_data_df = pd.read_csv('input_data.csv')
    # 文本向量化
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(new_data_df['cut_name'].values)
    print('不同词语个数：', len(tokenizer.word_index))
    # x = tokenizer.texts_to_sequences(new_data_df['cut_name'].values)
    # x = pad_sequences(x, maxlen=SP.MAX_LENGTH)
    # ========================
    word_index, weight_matrix = pre_matrix()
    x = create_tokenizer(new_data_df['cut_name'].values, word_index)
    # ========================
    y = pd.get_dummies(new_data_df['category3_new']).values
    # smote数据增强
    bl_smote = BorderlineSMOTE(k_neighbors=5, kind='borderline-1')  # ADASYN\SVMSMOTE\KMeansSMOTE
    x, y = bl_smote.fit_resample(x, y)
    # 拆分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.10, random_state=42)
    # 定义模型
    model = Sequential()
    # 参与学习
    # model.add(Embedding(SP.MAX_WORDS_NUM, output_dim=SP.EMBEDDING_DIM, input_length=X_train.shape[1]))
    # 迁移word2vec模型
    model.add(Embedding(input_dim=len(weight_matrix), output_dim=SP.EMBEDDING_DIM, input_length=X_train.shape[1],
                        weights=[weight_matrix], trainable=False))
    model.add(SpatialDropout1D(rate=0.2, seed=12))
    model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, ))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # 使用k折交叉验证
    kfold = KFold(n_splits=10)
    loss_list, accuracy_list = list(), list()
    k = 0
    for t_train, t_test in kfold.split(X_train, Y_train):
        print('============第{}折============'.format(k))
        k += 1
        model.fit(np.array(X_train[t_train]), np.array(Y_train[t_train]), epochs=5, batch_size=64, validation_split=0.1,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        accuracy = model.evaluate(np.array(X_train[t_test]), np.array(Y_train[t_test]))
        loss_list.append(round(accuracy[0], 3))
        accuracy_list.append(round(accuracy[1], 3))
    print('K-Loss: {}\n  K-Accuracy: {}'.format(loss_list, accuracy_list))
    print('Loss: {}\n  Accuracy: {}'.format(np.mean(loss_list), np.mean(accuracy_list)))
    # model.fit(X, Y, epochs=5, batch_size=64, validation_split=0.1,
    #           callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
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
    cat_id = pd.read_csv('../category_to_id.csv')
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
    # print(mod.layers[0].output[0])
    # predict_result(token, mod)
