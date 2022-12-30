import re
import jieba
import logging

def cut_word(word):
    out_word_list = []
    # 清洗特殊字符
    word = re.sub(r'\(.*?\)', '', word)
    word = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]|(丨)', '', word)
    # 形如:"EXO店x铺excelAxB" 去除x
    word = re.sub(r'(?<=[\u4e00-\u9fa5])(x|X)(?=[\u4e00-\u9fa5])|(?<=[A-B])x(?=[A-B])', '', word)
    # 设置不可分割的词
    with open('../no_cut_word.txt', 'r', encoding='utf-8') as in_word:
        for iw in in_word:
            iw = iw.strip('\n')
            jieba.suggest_freq(iw, True)
    l_cut_words = jieba.lcut(word)
    # 人工去除明显无用的词
    stop_words = [line.strip() for line in open('../useless_word.txt', 'r', encoding='utf-8').readlines()]
    for lc_word in l_cut_words:
        if lc_word not in stop_words:
            if lc_word != '\t':
                out_word_list.append(lc_word)
    return out_word_list
