from impala.dbapi import connect


# 对查询表数据量进行统计
def count_matching_number(ti_list, ta_list):
    conn = connect(host='124.71.220.115',  # 主机
                   port=10015,  # 端口
                   auth_mechanism='PLAIN',
                   user='hive',  # 用户
                   password='xwbigdata2022',
                   database='standard_db'  # 数据库
                   )
    cursor = conn.cursor()
    try:
        # 陈列信息表
        sql_yy = "select count(1) from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[0])
        sql_hn = "select count(1) from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[1], ta_list[2])
        sql_dl = "select count(1) from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[5])
        sql_jdb = "select count(1) from standard_db.{1} sd " \
                  "inner join standard_db.di_store_dedupe dedupe " \
                  "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[7])
        cursor.execute(sql_yy)
        display_yy_count = cursor.fetchall()
        cursor.execute(sql_hn)
        display_hn_count = cursor.fetchall()
        cursor.execute(sql_dl)
        display_dl_count = cursor.fetchall()
        cursor.execute(sql_jdb)
        display_jdb_count = cursor.fetchall()
        # 交易信息
        sql_hn = "select count(1) from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[4])
        sql_dl = "select count(1) from standard_db.{1} sd " \
                 "inner join standard_db.di_store_dedupe dedupe " \
                 "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ti_list[0], ta_list[6])
        cursor.execute(sql_hn)
        order_hn_count = cursor.fetchall()
        cursor.execute(sql_dl)
        order_dl_count = cursor.fetchall()
        # 拜访信息
        sql = "select count(distinct storeid) from standard_db.{0} sd " \
              "inner join standard_db.di_store_dedupe dedupe " \
              "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ta_list[8])
        cursor.execute(sql)
        visit_count = cursor.fetchall()
        # 其他表
        sql_hn1 = "select count(1) from standard_db.{0} sd " \
                  "inner join standard_db.di_store_dedupe dedupe " \
                  "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ta_list[1])
        sql_hn2 = "select count(1) from standard_db.{0} sd " \
                  "inner join standard_db.di_store_dedupe dedupe " \
                  "on dedupe.appcode <> '高德' and dedupe.appcode not like '%,%' and sd.storeid = dedupe.original_id" \
            .format(ta_list[3])
        cursor.execute(sql_hn1)
        activity_hn_count = cursor.fetchall()
        cursor.execute(sql_hn2)
        purchase_hn_count = cursor.fetchall()
        with open("statistics.txt", "w") as f:
            f.write("display_yy_count: " + str(display_yy_count) + "\r\n")
            f.write("display_hn_count: " + str(display_hn_count) + "\r\n")
            f.write("display_dl_count: " + str(display_dl_count) + "\r\n")
            f.write("display_jdb_count: " + str(display_jdb_count) + "\r\n")
            f.write("order_hn_count: " + str(order_hn_count) + "\r\n")
            f.write("order_dl_count: " + str(order_dl_count) + "\r\n")
            f.write("visit_count: " + str(visit_count) + "\r\n")
            f.write("activity_hn_count: " + str(activity_hn_count) + "\r\n")
            f.write("purchase_hn_count: " + str(purchase_hn_count) + "\r\n")
    except Exception:
        print("出错了！")
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
    count_matching_number(time_list, table_list)
