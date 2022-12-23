from ast import literal_eval

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB,BernoulliNB, MultinomialNB
from workplace.forone.count_category_num import count_the_number_of_categories


def test_transform():
    result = pd.DataFrame()
    csv_data = pd.read_csv('./aaa.csv', usecols=['name', 'category3_new', 'cut_name'])
    csv_data['cut_name'] = csv_data['cut_name'].apply(literal_eval)
    x = count_the_number_of_categories(csv_data)
    y = csv_data['category3_new']
    c_nb = ComplementNB()
    # transfer = TfidfTransformer()
    # x = transfer.fit_transform(x)
    print("拟合：：", x)
    c_nb.fit(x, y)
    print(c_nb.feature_log_prob_)


if __name__ == '__main__':
    test_transform()
