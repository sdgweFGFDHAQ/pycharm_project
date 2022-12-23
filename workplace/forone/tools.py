import re
import jieba


def cut_word(word):
    out_word_list = []
    # 加载停用词
    stop_words = [line.strip() for line in open('../stop_word_plug.txt', 'r', encoding='utf-8').readlines()]
    word = re.sub(r'\(.*?\)', '', word)
    word = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]|[丨]', '', word)
    # 形如:"EXO店x铺excelAxB" 去除x
    word = re.sub(r'(?<=[\u4e00-\u9fa5])(x|X)(?=[\u4e00-\u9fa5])|(?<=[A-B])x(?=[A-B])', '', word)
    # 不可分割的词
    with open('../inseparable_word_list.txt', 'r', encoding='utf-8') as in_word:
        for iw in in_word:
            iw = iw.strip('\n')
            jieba.suggest_freq(iw, True)
    l_cut_words = jieba.lcut(word)
    for lc_word in l_cut_words:
        if lc_word not in stop_words:
            if lc_word != '\t':
                out_word_list.append(lc_word)
    return out_word_list
