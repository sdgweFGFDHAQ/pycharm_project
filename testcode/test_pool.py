import os
from multiprocessing import Pool
import re
import jieba
# import torch
# import torchvision
import ast


def test():
    x = torch.empty(5, 3)
    print(x)
    print(torch.__version__)
    print(torchvision.__version__)
    print(torch.cuda.is_available())


# import torch


class Test(object):
    def __init__(self, st, stop_word):
        self.st = st
        self.stop_word = stop_word

    # def delete_sw(self):
    #     st_cuts = jieba.lcut(st)
    #     print(st_cuts)
    #     for st_cut in st_cuts:
    #         if stop_word.__contains__(st_cut):
    #             re.sub(st_cut, '已删除', st)

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
    # test()
    resub()
    # evaltest()