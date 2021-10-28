import os
import pyspark
import pandas as pd
from pyspark.sql.column import Column, _to_java_column, _to_seq
from pyspark.sql.types import IntegerType, LongType, StringType, DoubleType,ArrayType
from pyspark.sql.functions import udf
from utils.utils import CreateSparkContex

os.environ['PYSPARK_SUBMIT_ARGS'] = "--jars /Users/gallup/study/spark-test/target/spark-test.jar pyspark-shell"
sc, spark = CreateSparkContex()

spark.udf.registerJavaFunction("numAdd", "com.example.spark.AddNumber", LongType())
spark.udf.registerJavaFunction("numMultiply", "com.example.spark.MultiplyNumber", LongType())

hsy_data = {
    "label": [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0,  1],
    "keyword": ["安慕希", "牛奶", "牛", "奶粉", "婴儿奶粉", "液态奶", "牛肉", "奶", "牛肉干", "牛奶口味", "111大药房", "平安健康专项店", "鲁南大药房"],
    "title": ["安慕希", "牛奶", "牛", "奶粉", "婴儿奶粉", "液态奶", "牛肉", "奶", "牛肉干", "牛奶口味", "111大药房", "平安健康专项店", "鲁南大药房"],
    "brand": ["安慕希", "伊利", "蒙牛", "奶粉", "婴儿奶粉", "液态奶", "牛肉", "奶", "牛肉干", "牛奶口味", "999", "同仁堂", "仁和"],
    "tag": ["酸奶", "纯牛奶", "牛", "固态奶", "婴儿奶粉", "液态奶", "牛肉", "奶", "牛肉干", "牛奶口味", "药", "药", "药"],
    "volume": [1, 2, 3, 4, 5, 4.3, 1.2, 4.5, 1.0, 0.8, 0.6, 0.1, 0.2],
    "type": [0, 1, 0, 1, 2, 1, 0, 0, 2, 1, 0, 2, 1],
    # "spu_id": [39877457, 39877710, 39878084, 39878084, 39878084, 39877710, 39878084, 39877710, 39878084, 39878084],
    # "all_topic_fav_7": ["1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
    #                     "1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
    #                     "1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
    #                     "1: 0.4074,177: 0.1217,502: 0.4826",
    #                     "1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
    #                     "1: 0.4074,177: 0.1217,502: 0.4826"]

}

hsy_df = pd.DataFrame(hsy_data)
print(hsy_df.head(10))
hsy_df.to_csv('./spark_test.csv', index=False, sep='\t')
df = spark.read.format('csv').option('header', 'true').option('delimiter', '\t').option('inferSchema', 'true').load(
    "./spark_test.csv")
df.count()
df.show()
df.printSchema()
# df = df.withColumn('type',df['type'].cast(LongType()).alias('type'))
# df.registerTempTable("table")
# df1 = spark.sql("SELECT numMultiply(type) As num_1, numAdd(type) AS num_2 from table")
# df1.show(10)
#
#
#
# df.registerTempTable("table")
# spark.sql("SELECT SUM(volume) FROM table").show()
#
# print('test')
# spark.udf.registerJavaFunction("myCustomUdf", "com.example.spark.SparkJavaUdfExample", IntegerType())
# spark.sql("SELECT myCustomUdf(keyword) AS keyword_len from table").show()
#
# print('test 1')
# spark.udf.registerJavaFunction("segmentUdf", "com.example.spark.SegmentUdf", ArrayType(StringType()))
# spark.sql("SELECT segmentUdf(keyword) AS keyword_seg  from table").show()
# spark.sql("SELECT segmentUdf(title) AS title_seg  from table").show()
# spark.sql("SELECT segmentUdf(brand) AS brand_seg  from table").show()
#
# print('simi')
# spark.udf.registerJavaFunction("cosSimilarityUdf", "com.example.spark.CosSimilarity", DoubleType())
# query_sql = "SELECT cosSimilarityUdf({0},{1}) AS query_item_sim from table".format("keyword","title")
# print(query_sql)
# spark.sql(query_sql).show()
#
# def wordCount(r):
#     return len(r)
#
#
# wcUdf = udf(wordCount, IntegerType())
# spark.sql("SELECT myCustomUdf(keyword) from table").show()
#

def myCustomUdf(column):
    from pyspark.sql.column import Column, _to_java_column, _to_seq
    jc = spark._jvm.com.rootcss.SparkJavaUdfExample
    return Column(jc(_to_seq(sc, [column], _to_java_column)))
#

def my_udf(column):
    from pyspark.sql.column import Column, _to_java_column, _to_seq
    pcls = "com.example.spark.SparkJavaUdfExample"
    jc = sc._jvm.java.lang.Thread.currentThread() \
        .getContextClassLoader().loadClass(pcls).newInstance().getUdf().apply
    return Column(jc(_to_seq(sc, [column], _to_java_column)))

# def my_udf():
#     from pyspark.sql.column import Column, _to_java_column, _to_seq
#     pcls = "com.example.MyUdf"
#     jc = sc._jvm.java.lang.Thread.currentThread() \
#         .getContextClassLoader().loadClass(pcls).newInstance().getUdf().apply
#     return Column(jc(_to_seq(sc, [], _to_java_column)))

def myCol(col):
    _f = sc._jvm.com.example.spark.SparkJavaUdfExample.apply
    return Column(_f(_to_seq(sc,[col], _to_java_column)))

df.withColumn('_counter',my_udf('keyword')).show()
# df.withColumn('_counter', maturity_udf('text')).show()

#
# def myCol(col):
#     _f = sc._jvm.com.rootcss.SparkJavaUdfExample.apply
#     return Column(_f(_to_seq(sc,[col], _to_java_column)))
