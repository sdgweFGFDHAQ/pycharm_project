# -*- coding:utf-8 -*-
######################################################
# 描述：用于读取算法服务器csv文件写入di_store_dedupe_labeling
# 修改记录：
# 日期           版本       修改人    修改原因说明
# 2023/02/20     V1.00      lrz      新建代码
######################################################

import sys
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from recall_tools.pyspark_common import pySparkCm
import csv
from recall_tools.ssh import SSH
import gc

# 清空目标表数据
# table_name = 'di_store_store_visit_report'
# cm = pySparkCm(table_name)
# spark = cm.sparkenv()
# spark.sql("truncate table standard_db.di_store_store_visit_report")
# print("table truncate complete")

# 远程连接服务器读取文件
ssh = SSH()
connect = ssh.connect()
sftp_client = connect.open_sftp()
# 读取csv文件输出字典
data = []
try:
    with sftp_client.open("/home/data/temp/zzx/workbranch/data_score/store_visit_report.csv") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            dict_line = dict(row)
            data.append(dict_line)
except Exception as ex:
    sys.exit(ex)

sftp_client.close()
ssh.close()
print("read data success: " + str(len(data)))

# 定义变量
table_name = 'di_store_store_visit_report'
cm = pySparkCm(table_name)
spark = cm.sparkenv()

# store_id,report_json,assign_visitor,report_time,score,is_exist,assigner,is_examine,assign_time,examiner,examine_time
df = spark.createDataFrame(data) \
    .selectExpr("cast(store_id as long) as store_id", "report_json", "cast(assign_visitor as int) as assign_visitor",
                "report_time", "cast(score as int) as score", "is_exist", "assigner", "is_examine", "assign_time",
                "examiner", "examine_time")
df.printSchema()

# df = spark.read.option("header", "true").csv("/tmp/export/all.csv") \
#    .selectExpr("cast(store_id as long) as store_id","cast(same_id as long) as same_id","cast(l_same_id as int) as l_same_id","cast(a_same_id as int) as a_same_id","name","geo_point","address","adcode","state","city","district","name_cut","address_cut","appcode")


# 写入表
print("write data count=" + str(df.count()))
cm.write_to_hudi(df, 'standard_db', 'di_store_store_visit_report', 'store_id', '', 'store_id', 'append', 'insert')
print("Complete!")
