from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext


def CreateSparkContex():
    conf = SparkConf().setAppName('project1').setMaster('local').set("spark.jars",
                                                                     "/Users/gallup/study/search-ranking/config/spark-tfrecord_2.12-0.3.3_1.15.0.jar")
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return sc,spark
