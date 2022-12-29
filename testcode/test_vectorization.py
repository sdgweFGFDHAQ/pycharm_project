import time
from ast import literal_eval
import sys
import numpy
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, BernoulliNB, MultinomialNB
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from workplace.forone.count_category_num import feature_vectorization
from sklearn.feature_extraction.text import CountVectorizer
from workplace.forone.tools import cut_word

def test_transform():
    csv_data = pd.read_csv('./aaa.csv', usecols=['name', 'category3_new', 'cut_name'], nrows=5)
    csv_data['cut_name'] = csv_data['cut_name'].apply(literal_eval)
    lll = []
    for data in csv_data['cut_name']:
        list1 = [str(i) for i in data]
        str_data = ' '.join(list1)
        lll.append(str_data)
    print(lll)
    vectorized = CountVectorizer()
    transform = vectorized.fit_transform(lll)
    print(vectorized.get_feature_names_out())
    toarray = transform.toarray().astype(numpy.int8)
    s3 = time.localtime(time.time())
    # x = feature_vectorization(csv_data)
    # y = csv_data['category3_new']
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    # c_nb = ComplementNB()
    # transfer = TfidfTransformer()
    # x_train1 = transfer.fit_transform(x_train)
    # c_nb.fit(x_train1, y_train)
    # print(list(y_test))
    # print(c_nb.predict(x_test))
    # feature_prob = pd.DataFrame(c_nb.feature_log_prob_, index=c_nb.classes_, columns=x.columns)
    # print(feature_prob)


if __name__ == '__main__':
    test_transform()
