import sys
from pyspark.sql import SparkSession, HiveContext


#_SPARK_HOST = "spark://spark-master:7077"
_APP_NAME = "test"
spark = SparkSession.builder.enableHiveSupport().appName(_APP_NAME).getOrCreate()

# 使用拼接sql语句的方式查询hive 表,返回dataFrame格式数据
hive_database = "test"             #  要操作的数据库
hive_table = "table_01"            #  要操作的数据表
hive_read_sql = "select * from {}.{}".format(hive_database, hive_table)

read_df = spark.sql(hive_read_sql)
# hive_context = HiveContext(spark)
# hive_context.setLogLevel("WARN") # 或者INFO等
# read_df = hive_context.sql(hive_read_sql)
print(read_df.show(10))
print('*' * 10, '行为数据总量:', read_df.count(), ' 特征数量：', len(read_df.columns))
# 打印列名
print(read_df.columns)
print(read_df.printSchema())

# dataFrame 转化成rdd
rdd1 = read_df.rdd
print(rdd1.collect())
# 重复采样
data_p11 = rdd1.sample(True, samplerate, 100)


spark.stop()