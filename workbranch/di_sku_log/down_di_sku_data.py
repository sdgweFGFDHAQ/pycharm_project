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
        sql = "select dff.store_id, dff.sku_code,dff.brand_name,dff.series_name,dff.sku_name," \
                 "dff.category1_new, dff.category2_new,dff.category3_new,dff.name,dsdl.predict_category,dff.drink_label " \
                 "from (" \
                 "select ds.store_id, ds.sku_code,ds.brand_name,ds.series_name,ds.sku_name,ds.drink_label," \
                 "dsd.category1_new, dsd.category2_new,dsd.category3_new,dsd.name " \
                 "from (select ds.store_id, ds.sku_code,dssdl.brand_name," \
                 "dssdl.series_name,dssdl.sku_name,dssdl.drink_label " \
                 "from standard_db.di_store_sku_drink_label dssdl " \
                 "inner join standard_db.di_sku as ds on dssdl.sku_code = ds.sku_code) ds " \
                 "inner join standard_db.di_store_dedupe as dsd on dsd.appcode like '%高德%' and dsd.appcode like '%,%' and ds.store_id = dsd.id) dff " \
                 "left join standard_db.di_store_dedupe_labeling as dsdl on dff.store_id = dsdl.store_id" \
            .format()

        #合并去重
        # hbsql = "SELECT name, 'storeType', string_agg(DISTINCT drink_label, ',') AS drink_labels " \
        #         "FROM di_sku_log_drink_temp GROUP BY name, 'storeType' limit 10"

        cursor.execute(sql)
        di_sku_log_sql = cursor.fetchall()
        di_sku_log_data = as_pandas(di_sku_log_sql)
        di_sku_log_data.to_csv('di_sku_log_data.csv')
    except Exception:
        print("出错了！")
    finally:
        # 关闭连接
        cursor.close()
        conn.close()


if __name__ == '__main__':
    table_list = ['']
    count_matching_number(table_list)
