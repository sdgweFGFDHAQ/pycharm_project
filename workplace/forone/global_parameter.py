class StaticParameter:
    # 原始数据文件路径
    DATA_PATH = '../di_store_gz.csv'
    # 对店名分词字段
    CUT_NAME = 'cut_name'
    # 对分类有利的特征词最小出现次数
    MIN_NUMBER = 10
    # 对分类有利的特征词最大出现次数占总次数比例
    MAX_RATE = 0.75
    # 信息增益过低阈值
    LOW_IGR_PERCENT = 0.4
