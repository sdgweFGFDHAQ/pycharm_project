import re
import jieba
from torch.utils.data import Dataset


def set_jieba():
    # 设置不可分割的词
    jieba.load_userdict("../resources/stopwords.txt")
    jieba.load_userdict('../resources/indiv_words.txt')


def cut_word(word):
    # 清洗特殊字符
    word = re.sub(r'\(.*?\)|[^a-zA-Z0-9\u4e00-\u9fa5]|(丨)', ' ', str(word))
    # 形如:"EXO店x铺excelAxB" 去除x
    word = re.sub(r'(?<=[\u4e00-\u9fa5])([xX])(?=[\u4e00-\u9fa5])|(?<=[A-Z])x(?=[A-Z])', ' ', word)
    l_cut_words = jieba.lcut(word)
    # 人工去除明显无用的词
    out_word_list = list()
    stop_words = [line.strip() for line in open('../resources/stopwords.txt', 'r', encoding='utf-8').readlines()]
    for lc_word in l_cut_words:
        if lc_word not in stop_words:
            if lc_word != '\t' and not lc_word.isspace():
                out_word_list.append(lc_word)
    if out_word_list and (len(out_word_list) != 0):
        return ' '.join(out_word_list)
    else:
        return ' '.join(l_cut_words)


def error_callback(error):
    print(f"Error info: {error}")


class DefineDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
