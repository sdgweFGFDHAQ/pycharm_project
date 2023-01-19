from impala.util import as_pandas
from pyhive import hive


# 1获取目标表中，在di_store_dedupe表中存在且单源的数据df
def get_connection_all(ti_list, ta_list):
    #  192.168.0.150
    # 124.71.220.115
    conn = hive.Connection(host='192.168.0.150', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    cursor = conn.cursor()
    # cursor.arraysize = 1
    try:
        # sql1 = "select * from standard_db.di_store_visit_all limit 1"
        # sql2 = "select * from standard_db.di_store_dedupe " \
        #        "where appcode <> '高德' and appcode not like '%,%' limit 1"
        # cursor.execute(sql1)
        # fetchall1 = cursor.fetchall()
        # print(fetchall1)
        # cursor.execute(sql2)
        # fetchall2 = cursor.fetchall()
        # print(fetchall2)
        sql3 = "select count(distinct(appcode)) from standard_db.di_store_visit_all"
        sql4 = "select count(distinct(district)) from standard_db.di_store_dedupe"
        cursor.execute(sql3)
        fetchall3 = cursor.fetchall()
        print(fetchall3)
        cursor.execute(sql4)
        fetchall4 = cursor.fetchall()
        print(fetchall4)
        sql_city = "select distinct(appcode) from standard_db.di_store_visit_all"
        cursor.execute(sql_city)
        df = as_pandas(cursor)
        with open("appcode_list.txt", "w") as f:
            f.write(str(list(df['appcode'])))
        # sql5 = "show columns in standard_db.di_store_dedupe"
        # cursor.execute(sql5)
        # fetchall5 = cursor.fetchall()
        # print(fetchall5)
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
    get_connection_all(time_list, table_list)
