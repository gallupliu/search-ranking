from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.functions import *
from pyspark.sql.types import *
from collections import defaultdict
from pyspark.sql import functions as F


def CreateSparkContex():
    sparkconf = SparkConf().setAppName("MYPRO").set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=sparkconf)
    print("master:" + sc.master)
    sc.setLogLevel("WARN")
    spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()
    return sc, spark


def label(exposure_time, click_time, cart_time):
    if exposure_time is None:
        return 0
    else:
        if click_time is not None or cart_time is not None:
            return 1
        else:
            return 0


if __name__ == '__main__':
    sc, spark = CreateSparkContex()

    exposure_df = spark.createDataFrame([
        (1, "china", 72, 1, '2021-01-27 22:11:52'),
        (1, "china", 70, 4, '2021-01-27 22:11:52'),
        (1, "china", 79, 5, '2021-01-27 22:11:52'),
        (2, "usa", 78, 3, '2021-01-27 10:11:52'),
        (2, "usa", 48, 6, '2021-01-27 10:11:52'),
        (2, "usa", None, 6, '2021-01-27 10:11:52'),
        (2, "usa", 1000, 6, '2021-01-27 10:11:52'),
        (3, "\\N", 45, 3, '2021-01-27 11:11:10'),
        (3, "japan", 45, 3, '2021-01-27 11:11:10'),
        (3, "japan", 65, 5, '2021-01-27 11:11:10'), ],
        ['user_id', 'keyword', 'spu_id', 'position', 'exposure_time'])

    click_df = spark.createDataFrame([
        (1, "china", 72, 1, '2021-01-27 22:12:52'),
        (1, "china", 79, 5, '2021-01-27 22:13:22'),
        (1, "china", 79, 5, '2021-01-25 22:13:22'),
        (1, "china", 709, 5, '2021-01-27 22:13:22'),
        (2, "usa", 78, 3, '2021-01-27 10:12:12'),
        (2, "usa", 78, 3, '2021-01-27 10:10:12'),
        (2, "usa", 78, 3, None),
        (2, "china", 72, 3, '2021-01-27 10:12:12'), ],
        ['user_id', 'keyword', 'spu_id', 'position', 'click_time'])

    cart_df = spark.createDataFrame([
        (3, "japan", 45, '2021-01-27 11:12:10'),
        (2, "usa", 78, 3, '2021-01-25 10:12:12'),
        (2, "usa", 78, 3, '2021-01-27 10:13:12'),
        (2, "usa", 78, 3, None),
    ],
        ['user_id', 'keyword', 'spu_id', 'cart_time'])

    exposure_df.createOrReplaceTempView('exposure')
    click_df.createOrReplaceTempView('click')
    cart_df.createOrReplaceTempView('cart')
    exposure_df.show()
    click_df.show()
    cart_df.show()
    print(exposure_df.select('keyword').first()[0] )
    expr = "\\N"
    exposure_df = exposure_df.filter("spu_id is not null and spu_id != 1000")
    exposure_df.show()
    exposure_df = exposure_df.filter(exposure_df['keyword'] != '\\N')
    exposure_df.show()
    exposure_df.where(exposure_df.keyword.rlike("[0-9]")).show()
    # exposure_df.where("\\ not in (keyword)").show()
    keyword_df = spark.sql("select keyword,count(keyword)  as cnt from exposure   group by keyword order by cnt desc")
    keyword_df.show()
    merge_df = spark.sql("select \
                            a.user_id, \
                            a.keyword, \
                            a.spu_id,  \
                            a.exposure_time, \
                            b.click_time \
                      from exposure a left join click b on \
                       a.user_id = b.user_id and a.keyword = b.keyword and a.spu_id = b.spu_id")

    merge_df.createOrReplaceTempView('merge')
    merge_df = spark.sql("select \
                            a.user_id, \
                            a.keyword, \
                            a.spu_id,  \
                            a.exposure_time, \
                            a.click_time, \
                            b.cart_time \
                      from merge a left join cart b on \
                       a.user_id = b.user_id and a.keyword = b.keyword and a.spu_id = b.spu_id")
    merge_df.show()
    # merge_df.createOrReplaceTempView('merge')
    label_udf = udf(label, IntegerType())
    merge_df = merge_df.withColumn('label', label_udf('exposure_time', 'click_time', 'cart_time'))
    merge_df.show()

    merge_df.filter('label == 0').show()
    merge_df.filter('label == 1').show()