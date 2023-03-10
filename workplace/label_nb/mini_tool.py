import re
import jieba
import logging


def set_jieba():
    # 设置不可分割的词
    with open('../resources/indiv_words.txt', 'r', encoding='utf-8') as in_word:
        for iw in in_word:
            iw = iw.strip('\n')
            jieba.suggest_freq(iw, True)


def cut_word(word):
    out_word_list = []
    # 清洗特殊字符
    word = re.sub(r'\(.*?\)|[^a-zA-Z0-9\u4e00-\u9fa5]|(丨)', ' ', str(word))
    # 形如:"EXO店x铺excelAxB" 去除x
    word = re.sub(r'(?<=[\u4e00-\u9fa5])([xX])(?=[\u4e00-\u9fa5])|(?<=[A-Z])x(?=[A-Z])', ' ', word)
    l_cut_words = jieba.lcut(word)
    # 人工去除明显无用的词
    stop_words = [line.strip() for line in open('../resources/stopwords.txt', 'r', encoding='utf-8').readlines()]
    for lc_word in l_cut_words:
        if lc_word not in stop_words:
            if lc_word != '\t' and not lc_word.isspace():
                out_word_list.append(lc_word)
    return out_word_list


def error_callback(error):
    print(f"Error info: {error}")
