from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()

# Create DataFrame df1 with columns name,dept & age
data = [("James", "Sales", 34), ("Michael", "Sales", 56), \
        ("Robert", "Sales", 30), ("Maria", "Finance", 24)]
columns = ["name", "dept", "age"]
df1 = spark.createDataFrame(data=data, schema=columns)
df1.printSchema()
df1.show()

# Create DataFrame df1 with columns name,dep,state & salary
data2 = [("James", "Sales", "NY", 9000), ("Michael", "Sales", "CA", 9000), \
         ("Robert", "Sales", "NY", 7900), ("Maria", "Finance", "CA", 8000)]
columns2 = ["name", "dept", "state", "salary"]
# data2=[("James","Sales","NY",9000),("Maria","Finance","CA",9000), \
#     ("Jen","Finance","NY",7900),("Jeff","Marketing","CA",8000)]
df2 = spark.createDataFrame(data=data2, schema=columns2)
df2.printSchema()
df2.show()

# Add missing columns 'state' & 'salary' to df1
# from pyspark.sql.functions import lit
#
# for column in [column for column in df2.columns if column not in df1.columns]:
#     df1 = df1.withColumn(column, lit(None))
#
# df1.show()
#
# # Add missing column 'age' to df2
# for column in [column for column in df1.columns if column not in df2.columns]:
#     df2 = df2.withColumn(column, lit(None))
#
# df2.show()
# # Finally join two dataframe's df1 & df2 by name
# merged_df = df1.unionByName(df2)
# merged_df.show()

from pyspark.sql.types import StructType, StructField, LongType


def with_column_index(sdf):
    new_schema = StructType(sdf.schema.fields + [StructField("ColumnIndex", LongType(), False), ])
    return sdf.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(schema=new_schema)


print("join")
df1.join(df2, ['name', 'dept'], 'inner').show()
df1_ci = with_column_index(df1)
df2_ci = with_column_index(df2)
print('df1')
df1_ci.show()
df2_ci.show()
join_on_index = df1_ci.join(df2_ci, ['name', 'dept', "ColumnIndex"], 'inner').drop("ColumnIndex")
join_on_index.show()

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ProcessPoolExecutor

def test_job():
    print("python")

# 创建scheduler，多进程执行
executors = {
    'default': ProcessPoolExecutor(3)
}

scheduler = BlockingScheduler(executors=executors)
'''
 #该示例代码生成了一个BlockingScheduler调度器，使用了默认的默认的任务存储MemoryJobStore，以及默认的执行器ThreadPoolExecutor，并且最大线程数为10。
'''
scheduler.add_job(test_job, trigger='interval', seconds=5)
'''
 #该示例中的定时任务采用固定时间间隔（interval）的方式，每隔5秒钟执行一次。
 #并且还为该任务设置了一个任务id
'''
scheduler.start()
