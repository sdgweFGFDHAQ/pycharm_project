from impala.util import as_pandas
from pyhive import hive
import pandas as pd


# 1获取目标表中，在di_store_dedupe表中存在且单源的数据df
def get_connection_all(ti_list, ta_list, ac_list):
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
    try:
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
        for ac in ac_list:
            sql = "select storeid, {0} as createtime from standard_db.{1} sd " \
                  "inner join standard_db.di_store_dedupe dedupe " \
                  "on sd.appcode = '{2}' and dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' " \
                  "and sd.storeid = dedupe.original_id" \
                .format(ti_list[2], ta_list[8], ac)
            cursor.execute(sql)
            dl_order = as_pandas(cursor)
            dl_order.to_csv('visit_data.csv', mode='a')
        while True:
            data = cursor.fetchmany(size=50000)
            if len(data) == 0:
                break
            dl_order = as_pandas(data)
            dl_order.to_csv('visit_data.csv', mode='a')
    except Exception as e:
        print(str(e))
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
    appcode_list = ['雅士利国际集团有限公司', '养元', '江苏恒顺醋业股份有限公司', '卡士乳业（深圳）有限公司', '湖北贤哥食品有限公司',
                    '四川名扬永昌食品有限公司', '广州亿家馨食品有限公司', '大连林家铺子食品股份有限公司', '重庆有友食品销售有限公司',
                    '蜡笔小新（福建）食品工业有限公司', '九三食品', '广东鼎湖山泉有限公司', '上海妙可蓝多食品科技股份有限公司', '糊涂酒业',
                    '广州益力多乳品有限公司', '景田费用管理及资产管理系统V8.5', '加多宝', '好来化工（中山）有限公司',
                    '【正式环境】华彬快速消费品集团']
    get_connection_all(time_list, table_list, appcode_list)
