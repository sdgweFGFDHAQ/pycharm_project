# from impala.dbapi import connect
from pyhive import hive
import pandas as pd


# 对查询表数据量进行统计
def count_matching_number():
    # conn = hive.Connection(host='192.168.0.150', port=10015, username='hive', password='xwbigdata2022',
    #                        database='standard_db', auth='CUSTOM')
    conn = hive.Connection(host='124.71.220.115', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    cursor = conn.cursor()
    try:
        sql = "WITH sku AS (" \
              "select DISTINCT ds.store_id, dssdl.brand_name,dssdl.series_name,dssdl.sku_name,dssdl.sku_code,dssdl.drink_label " \
              "from standard_db.di_store_sku_drink_label dssdl " \
              "inner join standard_db.di_sku as ds " \
              "on dssdl.sku_name is not null and dssdl.sku_code = ds.sku_code) " \
              "select DISTINCT dsd.id as id,dsd.name as name,dsd.appcode as appcode,dsd.channeltype_new as channeltype_new," \
              "dsd.category1_new as category1_new,dsd.state as state,dsd.city as city, sku.brand_name as brand_name," \
              "sku.series_name as series_name,sku.sku_name as sku_name,sku.sku_code as sku_code,sku.drink_label as drink_label " \
              "from sku " \
              "inner join standard_db.di_store_dedupe as dsd on sku.store_id = dsd.id"

        cursor.execute(sql)
        di_sku_log_sql = cursor.fetchall()
        di_sku_log_data = pd.DataFrame(di_sku_log_sql,
                                       columns=["store_id", "sku_code", "brand_name", "series_name", "sku_name",
                                                "category1_new", "category2_new", "category3_new", "name",
                                                "predict_category", "drink_label"]).set_index("id")
        di_sku_log_data.to_csv('./data/di_sku_log_data_new.csv')
    except Exception as e:
        print("出错了！")
        print(e)
    finally:
        # 关闭连接
        cursor.close()
        conn.close()


if __name__ == '__main__':
    count_matching_number()

# nohup python -u down_drink_sku_data.py> down_di_sku_data.log 2>&1 &
