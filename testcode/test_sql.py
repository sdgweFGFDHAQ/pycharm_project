import os

from impala.util import as_pandas
from pyhive import hive
import pandas as pd


# 1获取目标表中，在di_store_dedupe表中存在且单源的数据df
def get_connection_all():
    #  192.168.0.150
    # 124.71.220.115
    conn = hive.Connection(host='192.168.0.150', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    cursor = conn.cursor()
    open('testdata.csv', 'w').close()
    try:
        # 拜访信息
        sql = "select dedupe.name from standard_db.di_store_visit_all sd " \
              "inner join standard_db.di_store_dedupe dedupe " \
              "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' " \
              "and sd.storeid = dedupe.original_id"
        cursor.execute(sql)
        while True:
            data = cursor.fetchmany(size=50000)
            if len(data) == 0:
                break
            print("3")
            dl_order = pd.DataFrame(data)
            print("4")
            if os.path.getsize('testdata.csv'):
                dl_order.to_csv('testdata.csv', mode='a', header=False, index=False)
            else:
                dl_order.to_csv('testdata.csv', mode='a', index=False)
            breakpoint()
        print('数据集收集完成：', 'testdata.csv')
    except Exception as e:
        print(str(e))
    finally:
        # 关闭连接
        cursor.close()
        conn.close()
        print('--连接已关闭--')


if __name__ == '__main__':
    # get_connection_all()

    cat_df = pd.read_csv('../workplace/label_lstm/category_to_id.csv')
    category_number = cat_df.shape[0]  # 78
    idx2lab = dict(zip(cat_df['cat_id'], cat_df['category1_new']))
    idx2lab[-1] = ''
