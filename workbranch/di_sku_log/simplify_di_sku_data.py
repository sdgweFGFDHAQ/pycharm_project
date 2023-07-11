# from impala.dbapi import connect
from pyhive import hive
from impala.util import as_pandas
import pandas as pd


# 对查询表数据量进行统计
def count_matching_number(ta_list):
    conn = hive.Connection(host='192.168.0.150', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    cursor = conn.cursor()
    try:
        # 合并去重
        # 同store_id
        sql = "SELECT store_id, name, 'storeType', string_agg(DISTINCT drink_label, ',') AS drink_labels " \
              "FROM di_sku_log_drink_temp GROUP BY store_id, name, 'storeType'"
        # 同name
        # sql = "SELECT name, 'storeType', string_agg(DISTINCT drink_label, ',') AS drink_labels " \
        #         "FROM di_sku_log_drink_temp GROUP BY name, 'storeType' limit 10"

        cursor.execute(sql)
        di_sku_log_sql = cursor.fetchall()
        di_sku_log_data = as_pandas(di_sku_log_sql)
        di_sku_log_data.to_csv('di_sku_log_single_drink_labels.csv')
    except Exception:
        print("出错了！")
    finally:
        # 关闭连接
        cursor.close()
        conn.close()


if __name__ == '__main__':
    table_list = ['']
    count_matching_number(table_list)
