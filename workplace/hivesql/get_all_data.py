from impala.dbapi import connect
from impala.util import as_pandas


# 1获取目标表中，在di_store_dedupe表中存在且单源的数据df
def get_connection_all(ti_list, ta_list):
    global cursor, conn
    try:
        #  192.168.0.150
        # 124.71.220.115
        conn = connect(host='192.168.0.150',  # 主机
                       port=10015,  # 端口
                       auth_mechanism='CUSTOM',
                       user='hive',  # 用户
                       password='xwbigdata2022',
                       database='standard_db'  # 数据库
                       )
        cursor = conn.cursor()
        # cursor.arraysize = 1
        # 陈列信息
        sql_yy = "select storeid, {0} from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[0])
        sql_hn = "select storeid, {0} as createname from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[1], ta_list[2])
        sql_dl = "select storeid, {0} from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[5])
        sql_jdb = "select storeid, {0} from standard_db.{1} sd " \
                  "inner join standard_db.di_store_dedupe dedupe " \
                  "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[7])
        sql = "(" + sql_yy + ") union all (" + sql_hn + ") union all (" + sql_dl + ") union all (" + sql_jdb + ")"
        cursor.execute(sql)
        yy_display = as_pandas(cursor)
        yy_display.to_csv('display_data.csv')
        # 交易信息
        sql_hn = "select storeid, {0} from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[4])
        sql_dl = "select storeid, {0} from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[6])
        sql = sql_hn + " union all " + sql_dl
        cursor.execute(sql)
        dl_order = as_pandas(cursor)
        dl_order.to_csv('order_data.csv')
        # 拜访信息
        sql = "select storeid, {0} as createname from standard_db.{1} sd " \
              "inner join standard_db.di_store_dedupe dedupe " \
              "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[2], ta_list[8])
        cursor.execute(sql)
        dl_order = as_pandas(cursor)
        dl_order.to_csv('visit_data.csv')
    except Exception:
        print("出现异常")
    finally:
        # 关闭连接
        cursor.close()
        conn.close()


if __name__ == '__main__':
    time_list = ['createtime', 'create_time', 'visitdate']
    table_list = ['di_store_yy_display',
                  'di_store_hn_activity', 'di_store_hn_display_project',
                  'di_store_hn_display_purchase', 'di_store_hn_receiving_note_v2',
                  'di_store_dl_display', 'di_store_dl_order',
                  'di_store_jdb_display',
                  'di_store_visit_all']
    get_connection_all(time_list, table_list)
