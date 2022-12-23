from ast import literal_eval

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB


def test_transform():
    result = pd.DataFrame()
    csv_data = pd.read_csv('./aaa.csv', usecols=['name', 'category3_new', 'cut_name'])
    csv_data['cut_name'] = csv_data['cut_name'].apply(literal_eval)
    x = csv_data['name']
    y = csv_data['category3_new']
    c_nb = ComplementNB()
    transfer = TfidfVectorizer()
    X = transfer.fit_transform(list(x))
    print(X)
    c_nb.fit(X, y)
    print(c_nb.predict(X))
    print("准确率为:", c_nb.score(X, y))


if __name__ == '__main__':
    test_transform()
