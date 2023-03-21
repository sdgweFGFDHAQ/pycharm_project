import re
import jieba


def set_jieba():
    # 设置不可分割的词
    # jieba.load_userdict("./resources/cityname.txt")
    # jieba.load_userdict("./resources/statename.txt")
    # jieba.load_userdict("./resources/distinctname.txt")
    # jieba.load_userdict("./resources/namenoise.txt")
    # jieba.load_userdict("./resources/symbol.txt")
    jieba.load_userdict("./resources/stopwords.txt")
    jieba.load_userdict('../resources/indiv_words.txt')


def cut_word(word):
    # 清洗特殊字符
    word = re.sub(r'\(.*?\)|[^a-zA-Z0-9\u4e00-\u9fa5]|(丨)', ' ', str(word))
    # 形如:"EXO店x铺excelAxB" 去除x
    word = re.sub(r'(?<=[\u4e00-\u9fa5])([xX])(?=[\u4e00-\u9fa5])|(?<=[A-Z])x(?=[A-Z])', ' ', word)
    l_cut_words = jieba.lcut(word)
    # 人工去除明显无用的词
    # out_word_list = list()
    # stop_words = [line.strip() for line in open('../resources/stopwords.txt', 'r', encoding='utf-8').readlines()]
    # for lc_word in l_cut_words:
    #     if lc_word not in stop_words:
    #         if lc_word != '\t' and not lc_word.isspace():
    #             out_word_list.append(lc_word)
    return ' '.join(l_cut_words)


def error_callback(error):
    print(f"Error info: {error}")
