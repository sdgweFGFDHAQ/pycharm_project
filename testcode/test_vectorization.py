import time
from ast import literal_eval
import sys
import numpy
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import ComplementNB, BernoulliNB, MultinomialNB
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from workplace.forone.count_category_num import feature_vectorization
from workplace.forone.mini_tool import cut_word


def test_transform():
    csv_data = pd.read_csv('./aaa.csv', usecols=['name', 'category3_new', 'cut_name'], nrows=5)
    csv_data['cut_name'] = csv_data['cut_name'].apply(literal_eval)
    lll = []
    for data in csv_data['cut_name']:
        list1 = [str(i) for i in data]
        str_data = ' '.join(list1)
        lll.append(str_data)
    print(lll)
    vectorized = TfidfVectorizer()
    transform = vectorized.fit_transform(lll)
    toarray = transform.toarray()
    print(toarray)
    s3 = time.localtime(time.time())


if __name__ == '__main__':
    test_transform()


def log_record():
    # 创建日志对象(不设置时默认名称为root)
    log = logging.getLogger()
    # 设置日志级别(默认为WARNING)
    log.setLevel('INFO')
    # 设置输出渠道(以文件方式输出需设置文件路径)
    file_handler = logging.FileHandler('test.log', encoding='utf-8')
    file_handler.setLevel('INFO')
    # 设置输出格式(实例化渠道)
    fmt_str = '%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt_str)
    # 绑定渠道的输出格式
    file_handler.setFormatter(formatter)
    # 绑定渠道到日志收集器
    log.addHandler(file_handler)
    return log
