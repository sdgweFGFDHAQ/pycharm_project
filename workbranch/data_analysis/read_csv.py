from pyhive import hive
import pandas as pd


# from impala.dbapi import connect
# from impala.util import as_pandas
# import sasl

def download_data():
    out_url = "/home/data/temp/zhouzx/workbranch/data_analysis/"
    conn = hive.Connection(host='192.168.0.150', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    # conn = hive.Connection(host='192.168.0.150',port=10015,username='ai',password='ai123456',
    #                      database='standard_db',auth='CUSTOM')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT category3_new from  standard_db.di_store_dedupe_labeling")
    category_list = cursor.fetchall()
    category_df = pd.DataFrame(category_list, columns=["category"])
    categories = category_df["category"].tolist()
    print(categories)
    print("已打标类别数量：", len(categories))
    data_count_list = []
    for category in categories:
        if category is None:
            continue
        print("开始执行sql")
        cursor.execute(
            "select count(1) from standard_db.di_store_dedupe_labeling where category3_new='" + category + "'")
        print("已经获取数据")
        data_count = cursor.fetchall()
        data_count_list.append(data_count[0][0])
    df = pd.DataFrame({'category': categories, 'count(1)': data_count_list})
    df.to_csv(out_url + "statistics_label.csv")
    print(df.head())
    print("统计完成")
    cursor.close()
    conn.close()


if __name__ == '__main__':
    download_data()
# nohup python -u main.py > log.log 2>&1 &
