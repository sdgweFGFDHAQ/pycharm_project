# from impala.dbapi import connect
from pyhive import hive
import pandas as pd


# 对查询表数据量进行统计
def count_matching_number(ta_list):
    conn = hive.Connection(host='192.168.0.150', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    cursor = conn.cursor()
    try:
        sql = "select ds.store_id, ds.sku_name,ds.drink_label,dsd.category1_new,dsd.name,dsd.state, dsd.appcode " \
              "from (" \
              "select ds.store_id, dssdl.sku_name,dssdl.drink_label " \
              "from standard_db.di_store_sku_drink_label dssdl " \
              "inner join standard_db.di_sku as ds " \
              "on dssdl.sku_code = ds.sku_code) ds inner " \
              "join standard_db.di_store_dedupe as dsd on ds.store_id = dsd.id"

        # 合并去重
        # hbsql = "SELECT name, 'storeType', string_agg(DISTINCT drink_label, ',') AS drink_labels " \
        #         "FROM di_sku_log_drink_temp GROUP BY name, 'storeType' limit 10"

        cursor.execute(sql)
        di_sku_log_sql = cursor.fetchall()
        di_sku_log_data = pd.DataFrame(di_sku_log_sql,
                                       columns=["store_id", "sku_code", "brand_name", "series_name", "sku_name",
                                                "category1_new", "category2_new", "category3_new", "name",
                                                "predict_category", "drink_label"]).set_index("id")
        di_sku_log_data.to_csv('di_sku_log_data_new.csv')
    except Exception as e:
        print("出错了！")
        print(e)
    finally:
        # 关闭连接
        cursor.close()
        conn.close()


if __name__ == '__main__':
    table_list = ['']
    count_matching_number(table_list)
