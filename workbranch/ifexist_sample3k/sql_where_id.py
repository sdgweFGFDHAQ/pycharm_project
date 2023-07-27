from pyhive import hive
import pandas as pd


def download_data():
    out_url = "/home/data/temp/zzx/"
    conn = hive.Connection(host='192.168.0.150', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    # conn = hive.Connection(host='192.168.0.150',port=10015,username='ai',password='ai123456',
    #                      database='standard_db',auth='CUSTOM')
    cursor = conn.cursor()
    city_list = ['广州市', '深圳市', '武汉市', '上海城区', '杭州市']
    samp = pd.DataFrame(columns=['id', 'name', 'namepath', 'city', 'district', 'township', 'address', 'geo_point', 'category1_new'])
    # 第一部分
    for city in city_list:
        print('执行sql' + city)
        if city is None:
            continue
        cursor.execute(
            "select id,name,namepath,city,district,township,address,geo_point,category1_new from standard_db.di_store_dedupe "
            "where appcode like '%高德%' and appcode like '%,%' and city=" + "'" +
            city + "'")
        data_list = cursor.fetchall()
        df = pd.DataFrame(data_list,
                          columns=['id', 'name', 'namepath', 'city', 'district', 'township', 'address', 'geo_point', 'category1_new'])
        df.to_csv(out_url + 'workbranch/ifexist_sample3k/' + city + '.csv',
                  columns=['id', 'name', 'namepath', 'city', 'district', 'township', 'address', 'geo_point', 'category1_new'], mode='w')
    # 第二部分
    cursor.execute(
        "select id,name,namepath,city,district,township,address,geo_point,category1_new "
        "from (select * from standard_db.di_store_dedupe WHERE city in ('广州市', '深圳市', '武汉市', '上海城区', '杭州市') "
        "and district in ('海珠区', '南山区', '武昌区', '浦东新区', '钱塘区')) dsd "
        "inner join standard_db.di_store_dedupe_ifexists dsdi on dsd.original_id=dsdi.store_id and dsdi.percentage>0.4 ")
    recall_data = cursor.fetchall()
    df = pd.DataFrame(recall_data,
                      columns=['id', 'name', 'namepath', 'city', 'district', 'township', 'address', 'geo_point', 'category1_new'])
    samp = pd.concat([df, samp], ignore_index=True)
    samp.to_csv(out_url + 'workbranch/ifexist_sample3k/ifexists_samples.csv',
                columns=['id', 'name', 'namepath', 'city', 'district', 'township', 'address', 'geo_point', 'category1_new'], mode='w')
    cursor.close()
    conn.close()


if __name__ == '__main__':
    download_data()
