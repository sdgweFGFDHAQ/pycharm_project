import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font", family='KaiTi')


def draw_picture():
    path = "result_df.csv"
    # 使用python下pandas库读取csv文件
    data = pd.read_csv(path, low_memory=False)
    # 读取列名为距离误差和时间点的所有行数据
    ydata = data.loc[:, 'percentage']
    xdata = list(data.index)
    # 读取列名为距离误差的前1000行数据
    # ydata = data.ix[:1000,'距离误差']
    # 点线图
    # plt.plot(xdata,ydata,'bo-',label=u'cte_误差',linewidth=1)
    # 点图
    plt.scatter(xdata, ydata, s=1)
    plt.title(u"权重置信率", size=10)
    plt.xlabel(u'门店召回数量', size=10)
    plt.ylabel(u'置信率（100%）', size=10)
    plt.show()


def statistics():
    df = pd.read_csv('result_df.csv')
    df_09 = df[df['percentage'] > 0.9]
    print('percentage>90:', df_09.index.size)
    df_08 = df[df['percentage'] > 0.8]
    print('percentage>80:', df_08.index.size)
    df_07 = df[df['percentage'] > 0.7]
    print('percentage>70:', df_07.index.size)
    df_06 = df[df['percentage'] > 0.6]
    print('percentage>60:', df_06.index.size)
    df_05 = df[df['percentage'] > 0.5]
    print('percentage>50:', df_05.index.size)
    df_04 = df[df['percentage'] > 0.4]
    print('percentage>40:', df_04.index.size)
    df_03 = df[df['percentage'] > 0.3]
    print('percentage>30:', df_03.index.size)
    df_02 = df[df['percentage'] > 0.2]
    print('percentage>20:', df_02.index.size)
    df_01 = df[df['percentage'] > 0.1]
    print('percentage>10:', df_01.index.size)
    df_000 = df[df['percentage'] > 0.00]
    print('percentage>00:', df_000.index.size)

    df_64 = df[df['percentage'] > 0.64]
    print('percentage>64:', df_64.index.size)


if __name__ == "__main__":
    draw_picture()
    statistics()
