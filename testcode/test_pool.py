import os
from multiprocessing import Pool, Manager
import re
import jieba
import ast


def resub():
    word = "发as|sdf35%^丨&*但是的风格aad$"
    word = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]|[丨]', '', word)
    print(word)
    word = "[dsgfd]g[dff]]"
    word = re.sub(r'\[|\]', '', word)
    print(word)
    word = "EXO是x的是X的excelAxB"
    word = re.sub(r'(?<=[\u4e00-\u9fa5])(x|X)(?=[\u4e00-\u9fa5])|(?<=[A-B])x(?=[A-B])', '', word)
    print(word)


def evaltest():
    value = "{'烤面包': 8.736542888491194, '系列': 9.142007996599357, '兰姐': 9.142007996599357, '旺城': 9.142007996599357,'莱': 9.142007996599357, '茂映': 9.142007996599357, '百家': 8.736542888491194, '林鲜生': 9.142007996599357,'Darling': 9.835155177159303, '升记': 9.142007996599357, '太鱼': 9.142007996599357, '客商': 9.142007996599357}"
    literal_eval = ast.literal_eval(value)
    print(literal_eval)


def pool_test():
    pool = Pool(processes=4)
    c = Manager().list()
    a = 1
    b = 2
    for i in range(4):
        pool.apply_async(co_num, args=(a, b, c))
    pool.close()
    pool.join()
    print(c)


def co_num(a, b, c):
    a_b = a + b
    c.append(a_b)
    return c


if __name__ == '__main__':
    # address_stopwords_set = {")", "(", "(", ")", "(", ")", "）", "（", "-", "号", "楼", "斋", "馆", "堂", "路",
    #                          "道", "街", "巷", "胡里", "条", "里", "省", "市", "层", "区", "县", "镇", "村",
    #                          "街道", "屯", "大街", "·", "米", "步行", "走", "交叉口", "约"}
    # print(address_stopwords_set)
    # s = ['中山', '八路', '97', '-', '101', '号', '(', '中山', '八', '地铁站', 'A', '口', '步行', '80', '米', ')']
    # for s1 in s:
    #     if address_stopwords_set.__contains__(s1):
    #         s.remove(s1)
    # print(s)
    # resub()
    # evaltest()
    pool_test()
