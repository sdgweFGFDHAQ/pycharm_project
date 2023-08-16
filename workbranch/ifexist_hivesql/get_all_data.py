import os

from impala.util import as_pandas
from pyhive import hive
import pandas as pd


# 1获取目标表中，在di_store_dedupe表中存在且单源的数据df
def get_connection_all(ti_list, ta_list):
    #  192.168.0.150
    # 124.71.220.115
    conn = hive.Connection(host='192.168.0.150', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    cursor = conn.cursor()
    # cursor.arraysize = 1

    sql_appcode = "SELECT DISTINCT(appcode) from standard_db.di_store_visit_all"
    cursor.execute(sql_appcode)
    app_list = cursor.fetchall()
    app_df = pd.DataFrame(app_list, columns=["appcode"])
    ac_list = app_df["appcode"].tolist()
    print('所有终端appcode类型:', ac_list)
    try:
        # 陈列信息
        sql_yy = "select dedupe.id, dedupe.name, dedupe.namepath, dedupe.township, dedupe.address, {0} from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode not like '%高德%'  and dedupe.namepath like '%广东%' " \
                 "and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[0])
        sql_hn = "select dedupe.id, dedupe.name, dedupe.namepath, dedupe.township, dedupe.address, {0} as createname from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode not like '%高德%'  and dedupe.namepath like '%广东%' " \
                 "and sd.storeid = dedupe.original_id" \
            .format(ti_list[1], ta_list[2])
        sql_dl = "select dedupe.id, dedupe.name, dedupe.namepath, dedupe.township, dedupe.address, {0} from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode not like '%高德%'  and dedupe.namepath like '%广东%' " \
                 "and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[5])
        sql_jdb = "select dedupe.id, dedupe.name, dedupe.namepath, dedupe.township, dedupe.address, {0} from standard_db.{1} sd " \
                  "inner join standard_db.di_store_dedupe dedupe " \
                  "on dedupe.appcode not like '%高德%'  and dedupe.namepath like '%广东%' " \
                  "and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[7])
        sql = "(" + sql_yy + ") union all (" + sql_hn + ") union all (" + sql_dl + ") union all (" + sql_jdb + ")"
        cursor.execute(sql)
        yy_display = as_pandas(cursor)
        yy_display.to_csv('display_data.csv', index=False)
        print('数据集收集完成：', 'display_data.csv')
        # 交易信息
        sql_hn = "select dedupe.id, dedupe.name, dedupe.namepath, dedupe.township, dedupe.address, {0} from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode not like '%高德%'  and dedupe.namepath like '%广东%' " \
                 "and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[4])
        sql_dl = "select dedupe.id, dedupe.name, dedupe.namepath, dedupe.township, dedupe.address, {0} from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode not like '%高德%'  and dedupe.namepath like '%广东%' " \
                 "and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[6])
        sql = sql_hn + " union all " + sql_dl
        cursor.execute(sql)
        dl_order = as_pandas(cursor)
        dl_order.to_csv('order_data.csv', index=False)
        print('数据集收集完成：', 'order_data.csv')
        # 拜访信息
        open('visit_data.csv', 'w').close()
        sql = "select dedupe.id, dedupe.name, dedupe.namepath, dedupe.township, dedupe.address, {0} as createtime from standard_db.{1} sd " \
              "inner join standard_db.di_store_dedupe dedupe " \
              "on on dedupe.appcode not like '%高德%'and dedupe.namepath like '%广东%' " \
              "and sd.storeid = dedupe.original_id" \
            .format(ti_list[2], ta_list[8])
        cursor.execute(sql)
        all_visit = as_pandas(cursor)
        all_visit.to_csv('visit_data.csv', index=False)
        # while True:
        #     data = cursor.fetchmany(size=50000)
        #     if len(data) == 0:
        #         break
        #     dl_order = as_pandas(data)
        #     if os.path.getsize('visit_data.csv'):
        #         dl_order.to_csv('visit_data.csv', mode='a', header=False, index=False)
        #     else:
        #         dl_order.to_csv('visit_data.csv', mode='a', index=False)
        print('数据集收集完成：', 'visit_data.csv')
    except Exception as e:
        print(str(e))
    finally:
        # 关闭连接
        cursor.close()
        conn.close()
        print('--连接已关闭--')


if __name__ == '__main__':
    time_list = ['createtime', 'create_time', 'visitdate']
    table_list = ['di_store_yy_display',
                  'di_store_hn_activity', 'di_store_hn_display_project',
                  'di_store_hn_display_purchase', 'di_store_hn_receiving_note_v2',
                  'di_store_dl_display', 'di_store_dl_order',
                  'di_store_jdb_display',
                  'di_store_visit_all']
    get_connection_all(time_list, table_list)
