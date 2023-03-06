import re
import jieba
import logging


def set_jieba():
    # 设置不可分割的词
    with open('../indiv_words.txt', 'r', encoding='utf-8') as in_word:
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
    stop_words = [line.strip() for line in open('../stopwords.txt', 'r', encoding='utf-8').readlines()]
    for lc_word in l_cut_words:
        if lc_word not in stop_words:
            if lc_word != '\t' and not lc_word.isspace():
                out_word_list.append(lc_word)
    return out_word_list


def error_callback(error):
    print(f"Error info: {error}")


# def log_record():
#     # 创建日志对象(不设置时默认名称为root)
#     log = logging.getLogger()
#     # 设置日志级别(默认为WARNING)
#     log.setLevel('INFO')
#     # 设置输出渠道(以文件方式输出需设置文件路径)
#     file_handler = logging.FileHandler('test.log', encoding='utf-8')
#     file_handler.setLevel('INFO')
#     # 设置输出格式(实例化渠道)
#     fmt_str = '%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
#     formatter = logging.Formatter(fmt_str)
#     # 绑定渠道的输出格式
#     file_handler.setFormatter(formatter)
#     # 绑定渠道到日志收集器
#     log.addHandler(file_handler)
#     return log

# def get_all():
#     standard_df = pd.DataFrame(columns=['id', 'name', 'category3_new', 'cut_name'])
#     for i in range(SP.SEGMENT_NUMBER):
#         path = SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(i) + '.csv'
#         df_i = pd.read_csv(path, usecols=['id', 'name', 'category3_new', 'cut_name'], keep_default_na=False)
#         df_i = df_i[df_i['category3_new'] != '']
#         standard_df_i = df_i.sample(frac=1)
#         standard_df = pd.concat([standard_df, standard_df_i])
#         standard_df = standard_df.sample(frac=1).reset_index(drop=True)
#     standard_df.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'all_labeled_data.csv', index=False)