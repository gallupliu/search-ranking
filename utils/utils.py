from pyspark import SparkConf,SparkContext
def CreateSparkContex():
    conf = SparkConf().setAppName('project1').setMaster('local')
    sc = SparkContext.getOrCreate(conf)
    return sc