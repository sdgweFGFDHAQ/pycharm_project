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
    # 陈列信息表
    sql_yy = "select count(1) from standard_db.{1} " \
             "where storeid in (select original_id from standard_db.di_store_dedupe where appcode not like '%,%') " \
        .format(ti_list[0], ta_list[0])
    sql_hn = "select count(1) as createname from standard_db.{1} " \
             "where storeid in (select original_id from standard_db.di_store_dedupe where appcode not like '%,%') " \
        .format(ti_list[1], ta_list[2])
    sql_dl = "select count(1) from standard_db.{1} " \
             "where storeid in (select original_id from standard_db.di_store_dedupe where appcode not like '%,%') " \
        .format(ti_list[0], ta_list[5])
    sql_jdb = "select count(1) from standard_db.{1} " \
              "where storeid in (select original_id from standard_db.di_store_dedupe where appcode not like '%,%') " \
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
    sql_hn = "select count(1) from standard_db.{1} " \
             "where storeid in (select original_id from standard_db.di_store_dedupe where appcode not like '%,%') " \
        .format(ti_list[0], ta_list[4])
    sql_dl = "select count(1) from standard_db.{1} " \
             "where storeid in (select original_id from standard_db.di_store_dedupe where appcode not like '%,%') " \
        .format(ti_list[0], ta_list[6])
    cursor.execute(sql_hn)
    order_hn_count = cursor.fetchall()
    cursor.execute(sql_dl)
    order_dl_count = cursor.fetchall()
    # 拜访信息
    sql = "select count(1) as createname from standard_db.{1} " \
          "where storeid in (select original_id from standard_db.di_store_dedupe where appcode not like '%,%') " \
        .format(ti_list[2], ta_list[8])
    cursor.execute(sql)
    visit_count = cursor.fetchall()
    with open("statistics.txt", "w") as f:
        f.write("display_yy_count: " + str(display_yy_count) + "\r\n")
        f.write("display_hn_count: " + str(display_hn_count) + "\r\n")
        f.write("display_dl_count: " + str(display_dl_count) + "\r\n")
        f.write("display_jdb_count: " + str(display_jdb_count) + "\r\n")
        f.write("order_hn_count: " + str(order_hn_count) + "\r\n")
        f.write("order_dl_count: " + str(order_dl_count) + "\r\n")
        f.write("visit_count: " + str(visit_count) + "\r\n")


if __name__ == '__main__':
    time_list = ['createtime', 'create_time', 'visitdate']
    table_list = ['di_store_yy_display',
                  'di_store_hn_activity', 'di_store_hn_display_project',
                  'di_store_hn_display_purchase', 'di_store_hn_receiving_note_v2',
                  'di_store_dl_display', 'di_store_dl_order',
                  'di_store_jdb_display',
                  'di_store_visit_all']
    count_matching_number(time_list, table_list)
