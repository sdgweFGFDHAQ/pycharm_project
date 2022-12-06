import os
from multiprocessing import Pool
import re
import jieba
import torch
import torchvision


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

    def delete_sw(self):
        st_cuts = jieba.lcut(st)
        print(st_cuts)
        for st_cut in st_cuts:
            if stop_word.__contains__(st_cut):
                re.sub(st_cut, '已删除', st)


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
    test()
