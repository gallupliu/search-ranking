from pyspark.sql.functions import to_date

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.functions import *
from pyspark.sql.types import *
from collections import defaultdict
from pyspark.sql import functions as pyf
import time
import datetime as dt
from pyspark.sql.functions import col, collect_list
from pyspark.sql.types import StringType


def CreateSparkContex():
    sparkconf = SparkConf().setAppName("MYPRO").set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=sparkconf)
    print("master:" + sc.master)
    sc.setLogLevel("WARN")
    spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()
    return sc, spark


def drop_ambigus01(df, columns):
    """去掉有冲突的0 1 标签"""
    print("原始数据:{0}".format(df.count()))
    df0 = df.filter('label==0').dropDuplicates(columns)
    df1 = df.filter('label==1').dropDuplicates(columns)

    df3 = df1.select(*columns).intersect(df0.select(*columns))
    print("交集个数：{0}".format(df3.count()))
    df3.show()
    df_drop01 = df.dropDuplicates(columns).select(*columns).subtract(df3.select(*columns))
    df_drop01.show()
    result_df = df.join(df_drop01, on=columns, how='right')
    print("merge")
    result_df.show()
    return result_df


def drop_ambigus0(df, columns):
    """去掉有冲突的0标签"""
    print("原始数据:{0}".format(df.count()))
    df0 = df.filter('label==0').dropDuplicates(columns)
    df1 = df.filter('label==1').dropDuplicates(columns)
    df3 = df1.select(*columns).intersect(df0.select(*columns))
    print('df_drop0')
    df3.show()

    tmp_df = df.join(df3, on=columns, how='right').filter('label==0').select(*(columns+["label"]))
    df_drop0 = df.select(*(columns+["label"])).subtract(tmp_df)
    df_drop0.show()

    result_df = df.join(df_drop0,on=columns+["label"],how='right')
    print('result_df')
    result_df.show()
    return result_df


sc, spark = CreateSparkContex()
#
# cart_df = spark.createDataFrame([
#     (3, "japan", 4, '2021-01-27 11:12:10', '中国飞鹤', '奶粉'),
#     (3, "japan", 40, '2021-01-27 11:12:10', '北京市', '北京'),
#     (3, "japan", 45, '2021-01-27 11:12:10', '北京', '北京'),
#     (3, "japan", 4, '2020-11-27 11:12:10', 'KN95口罩', '口罩'),
#     (3, "japan", 40, '2020-11-27 11:12:10', '成人口罩', '口罩'),
#     (3, "japan", 45, '2020-11-27 11:12:10', '内分泌', '奶粉'),
#     (2, "usa", 7, '2020-10-25 10:12:12', '杭州', '安慕希'),
#     (2, "usa", 70, '2020-10-25 10:12:12', '安慕希', '牛奶'),
#     (2, "usa", 8, '2020-10-27 10:13:12', '酸奶', '牛奶'),
#     (2, "usa", 7, '2021-01-25 10:12:12', '牛奶', '牛奶'),
#     (2, "usa", 70, '2021-01-25 10:12:12', '纯牛奶', '牛奶'),
#     (2, "usa", 8, '2021-01-27 10:13:12', '中国飞鹤', '奶粉'),
#     (2, "usa", 78, '2021-01-27 10:13:12', '中国飞鹤', '奶粉'),
#     (2, "usa", 78, None, '中国飞鹤', '奶粉'),
#     (2, "usa", 18, None, '中国飞鹤', '奶粉'),
# ],
#     ['user_id', 'keyword', 'spu_id', 'cart_time', 'title', 'category'])
#
# cart_df.show()
# # Following code line converts String into Date format
# print('fill')
# cart_df.show()
# # print(dt.datetime(1900, 1, 1))
# # cart_df
# cart_df = cart_df.withColumn('new_date',
#                              when(col('cart_time').isNull(), dt.datetime(1900, 1, 1)).otherwise(col('cart_time')))
# cart_df.show()
# cart_df = cart_df.withColumn('month', pyf.date_format(pyf.col('new_date'), 'yyyy-MM'))
# cart_df.write.option("header", "true").mode("overwrite").csv('./train.csv')
# cart_df.show()
# cart_df.filter("month='2020-11'").show()
# month_df = cart_df.groupBy('month').count().show()
#
#
# def array_to_string(my_list):
#     return '[' + ','.join([str(elem) for elem in my_list]) + ']'
#
#
# array_to_string_udf = udf(array_to_string, StringType())
#
# # df.groupBy("id").agg(collect_list("fName"), collect_list("lName"))
# session_df = cart_df.groupBy(['user_id', 'keyword', 'new_date', 'title', 'category']).agg(
#     collect_list('spu_id').alias('newcol'))
# print('session')
# session_df = session_df.withColumn('column_as_str', array_to_string_udf(session_df["newcol"]))
# print('string session')
# session_df.show()
#
# session_df.drop("newcol").write.option("header", "true").mode("overwrite").csv('./session.csv')
# # place all data in a single partition
# session_df.drop("newcol").coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").mode(
#     "overwrite").save(
#     "mydata.csv")
#
#
# def load(files, columns):
#     df = spark.read.format('csv') \
#         .option('header', 'true') \
#         .option('delimiter', ',') \
#         .load(files)
#     # for c, n in zip(df.columns,columns):
#     #     df.withColumnRenamed(c,n)
#     return df
#
#
# def text_data_process(df, column):
#     def get_char(text):
#         """char"""
#         chars = []
#         for char in text:
#             chars.append(char)
#         return ' '.join(chars)
#
#     getCharsUdf = pyf.udf(get_char, StringType())
#     df = df.withColumn(column + '_char', getCharsUdf(column))
#     return df
#
#
# def id_data_process(df):
#     df.select('column_as_str').coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").mode(
#         "overwrite").save(
#         "id.csv")
#
#
# test_df = load('./mydata.csv', [])
# id_data_process(test_df)
# test_df = text_data_process(test_df,'title')
# test_df = text_data_process(test_df,'category')
# test_df = test_df.select(['title_char','category_char'])
# test_df = test_df.withColumn('chars',pyf.concat_ws(' ',col('title_char'),col('category_char')))
columns = ['title_char', 'category_char']

# test_df = test_df.withColumn('chars',pyf.concat_ws(' ','title_char','category_char'))
# test_df.show()
# test_df.drop("newcol").coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").mode(
#     "overwrite").save(
#     "char.csv")
# test_df.show()
# test_df = test_df.withColumn('raws', pyf.concat_ws(' ', 'title', 'category'))
# test_df.show()
# test_df.drop("newcol").coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").mode(
#     "overwrite").save(
#     "raw.csv")
# print('convert')

df = spark.createDataFrame([
    ("key", "A", "a", 1, 0,'2021-01-27 11:12:10'), ("key", "A","b", 0, 2, '2021-01-27 11:12:10'),
    ("key", "A", "c", 1, 10,'2021-01-27 11:12:10'), ("key", "A", "d", 0, 11,'2021-01-27 11:12:10'),
    ("key", "B", "a", 1, 0,'2021-01-27 11:12:10'), ("key", "B", "b",1, 2,'2021-01-27 11:12:10'),
    ("key", "B", "c", 0,10, '2021-01-27 11:12:10'), ("key", "B", "d", 0,11, '2021-01-27 11:12:10'),
                            ("key", "C", "a", 1, 0,'2021-02-27 11:12:10'), ("key", "C", "b", 1, 2,'2021-02-27 11:12:10'),
                            ("key", "C", "c", 1,10, '2021-02-27 11:12:10'), ("key", "C", "d", 0, 11,'2021-02-27 11:12:10')
                            ],
                           ["query", "userid","spuid", "label",  "feature","time"])

df.show()
# pos_df = df.select(*["query", "spuid", "label"]).distinct().filter("label == 1").withColumnRenamed("label", "pos_label")
# pos_df.show()
# new_df = df.join(pos_df, ["query", "spuid"], "left").withColumn("new_label",
#                                                                 pyf.when(col('pos_label').isNull(), 0).otherwise(
#                                                                     col('pos_label'))).drop(
#     "pos_label").withColumnRenamed("new_label", "pos_label")
# new_df.show()
# new_df_1 = new_df.filter("label == pos_label or (pos_label == null and label == 0)").orderBy(
#     ["query", "userid", "spuid"])
# print('去掉冲突的0')
# new_df_1.show()

print('去掉冲突的0和1')
# df.groupBy(["query", "userid", "spuid"]).agg(pyf.collect_list('spuid').alias('seq'))
save_columns = ["query", "spuid"]
#
# df3 = df1.select(*save_columns).intersect(df0.select(*save_columns))
# print("交集个数：{0}".format(df3.count()))
# df3.show()
# df4 = df.select(*save_columns).subtract(df3.select(*save_columns))
# df4.show()
# df_drop01 = df.dropDuplicates(save_columns).select(*save_columns).subtract(df3.select(*save_columns))
# result_2 = df.join(df_drop01,on=save_columns,how='right')
print("0 1")
# result_df = drop_ambigus01(df, save_columns)
# result_df.show()

print("0")
result_df = drop_ambigus0(df, save_columns)
# new_df_2 = new_df.filter("label == pos_label or (pos_label == null and label == 0)")
# new_df_2.show()
# zero_df = df.filter("label == 0")
# one_df = df.filter("label == 1")
# zero_df.show()
# one_df.show()
# one_df.subtract(zero_df).show()
# count_feature_columns = ["rate", "price", "uv_rate"]
# df = df.withColumn("count_feature", pyf.to_str(pyf.struct([df[x] for i,x in enumerate(count_feature_columns)])))
# df = df.withColumn("count_feature",
#                    pyf.to_json(pyf.struct([df[x].alias(str(i)) for i, x in enumerate(count_feature_columns)])))
#
#
# def remove(column):
#     return column.replace('{', '').replace('}', '')
#
#
# removeUDf = udf(remove, StringType())
# df = df.withColumn('count_feature',removeUDf("count_feature"))

# df.select("count_feature").show()
# df = df.withColumn("rate", df["rate"].cast(DoubleType()).alias("rate"))
# df.show()
# df.printSchema()
# df.groupby("id").agg(pyf.collect_set("code"), pyf.collect_list("name")).show()

# cart_df = cart_df.withColumn('date',F.date_format(cart_df['arrival_date'],'yyyy-MM-dd HH:mm:ss'))
# cart_df = cart_df.withColumn('new_date', when(cart_df['cart_time'].isNull(), F.date_format('1900-01-01 00:00:00','yyyy-MM-dd HH:mm:ss')).otherwise(cart_df['cart_time']))
# cart_df.show()

# new_cart_df = spark.sql("""select """)
# default_time = '2020-01-01 00:00:10'
# time_array = time.strptime(default_time,'%Y-%m-%d %H:%M:%S')
# print(time_array)
# timestamp = int(time.mktime(time_array))
# print(timestamp)

# print('test')
# cart_df = cart_df.withColumn("Timestamp", (col("cart_time").cast("timestamp")))
# cart_df.show()
# df = spark.createDataFrame([(dt(2020,12,27),)], ['t'])
# df.show()
# df1 = df.select(to_date(df.t, 'yyyy-MM-dd').alias('dt')).withColumn('fn1',date_format(col('dt'),'YYYYMMdd'))
# df1.show()


from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors

dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0]),),
    (1, Vectors.dense([2.0]),),
    (2, Vectors.dense([3.0]),)
], ["id", "features"])

dataFrame.show()
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics and generate MinMaxScalerModel
scalerModel = scaler.fit(dataFrame)

# rescale each feature to range [min, max].
scaledData = scalerModel.transform(dataFrame)
firstelement=pyf.udf(lambda v:float(v[0]),FloatType())
print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
scaledData.select("features", "scaledFeatures").show()

scaledData.withColumn("scaledFeatures", firstelement("scaledFeatures")).show()