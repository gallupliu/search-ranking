from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import time
def print_run_time(func):
    """ 计算时间函数
    """

    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print('Current function : {function}, time used : {temps}'.format(
            function=func.__name__, temps=time.time() - local_time)
        )
        return res

    return wrapper
def CreateSparkContex():
    conf = SparkConf().setAppName('project1').setMaster('local').set("spark.jars",
                                                                     "/Users/gallup/work/java/spark-tfrecord/target/spark-tfrecord_2.12-0.3.3.jar")
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return sc,spark
