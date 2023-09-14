import pandas as pd

csv_path = {0: './datasets/sku_predict_result2.csv',
            1: './datasets/sku_predict_result2.csv',
            2: './datasets/sku_predict_result2.csv'}
csv_columns = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料', '碳酸饮料', '咖啡（类）饮料',
               '包装饮用水', '特殊用途饮料']


def get_vote_result():
    # base_csv:用于基模型预测 meta_csv:元模型验证
    proto_df = pd.read_csv(csv_path[0], usecols=csv_columns)
    sim_df = pd.read_csv(csv_path[1], usecols=csv_columns)
    textcnn_df = pd.read_csv(csv_path[2], usecols=csv_columns)
    sum_df = proto_df + sim_df + textcnn_df
    print(sum_df.head())

    vote_df = pd.DataFrame(columns=csv_columns)
    for column in csv_columns:
        vote_series = sum_df[column].apply(lambda x: 1 if x > 1.5 else 0)
        vote_df[column] = vote_series
    print(vote_df.head())


def get_mean_result(weights=None):
    if weights is None:
        weights = [1 / 3, 1 / 3, 1 / 3]
    r_p, r_s, r_t = weights
    # base_csv:用于基模型预测 meta_csv:元模型验证
    proto_df = pd.read_csv(csv_path[0], usecols=csv_columns)
    sim_df = pd.read_csv(csv_path[1], usecols=csv_columns)
    textcnn_df = pd.read_csv(csv_path[2], usecols=csv_columns)
    sum_df = r_p * proto_df + r_s * sim_df + r_t * textcnn_df
    print(sum_df.head())

    vote_df = pd.DataFrame(columns=csv_columns)
    for column in csv_columns:
        vote_series = sum_df[column].apply(lambda x: 1 if x > 0.5 else 0)
        vote_df[column] = vote_series
    print(vote_df.head())


if __name__ == '__main__':
    # 用于基模型预测及元模型验证
    # get_vote_result()  # 投票法
    get_mean_result()  # 加权平均
