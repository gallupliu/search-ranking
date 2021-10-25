from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from utils.utils import CreateSparkContex

path = "test-output.tfrecord"

# label	keyword	title	brand	tag	volume	type
# fields = [StructField("id", IntegerType()), StructField("IntegerCol", IntegerType()),
#           StructField("LongCol", LongType()), StructField("FloatCol", FloatType()),
#           StructField("DoubleCol", DoubleType()), StructField("VectorCol", ArrayType(DoubleType(), True)),
#           StructField("StringCol", StringType())]
# test_rows = [[11, 1, 23, 10.0, 14.0, [1.0, 2.0], "r1"], [21, 2, 24, 12.0, 15.0, [2.0, 2.0], "r2"]]
fields = [StructField("id", IntegerType()), StructField("keyword", StringType()),
          StructField("title", StringType()), StructField("brand", StringType()),
          StructField("tag", StringType()),
          StructField("volume", FloatType()),
          StructField("type", IntegerType()),
          StructField("user_bert_emb", ArrayType(DoubleType(), True)),
          StructField("item_bert_emb", ArrayType(DoubleType(), True)),
          StructField("label", IntegerType())]
schema = StructType(fields)
hsy_data = {
    "label": [0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    "keyword": ["安 慕 希", "牛 奶", "牛", "奶 粉", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
    "title": ["安 慕 希", "牛 奶", "牛", "奶 粉", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
    "brand": ["安 慕 希", "伊 利", "蒙 牛", "奶 粉", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
    "tag": ["酸 奶", "纯 牛 奶", "牛", "固 态 奶", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
    "volume": [1, 2, 3, 4, 5, 4.3, 1.2, 4.5, 1.0, 0.8],
    "type": [0, 1, 0, 1, 2, 1, 0, 0, 2, 1],
    # "spu_id": [39877457, 39877710, 39878084, 39878084, 39878084, 39877710, 39878084, 39877710, 39878084, 39878084],
    # "all_topic_fav_7": ["1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
    #                     "1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
    #                     "1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
    #                     "1: 0.4074,177: 0.1217,502: 0.4826",
    #                     "1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
    #                     "1: 0.4074,177: 0.1217,502: 0.4826"]

}
test_rows = []
for i in range(len(hsy_data["label"])):
    test_row = []
    test_row.append(i)
    test_row.append(hsy_data["keyword"][i])
    test_row.append(hsy_data["title"][i])
    test_row.append(hsy_data["brand"][i])
    test_row.append(hsy_data["tag"][i])
    test_row.append(float(hsy_data["volume"][i]))
    test_row.append(hsy_data["type"][i])
    test_row.append([0.0, 0.1, 0.5, 0.1, 0.5])
    test_row.append([0.1, 0.0, 0.2, 0.4, 0.5])
    test_row.append(hsy_data["label"][i])
    test_rows.append(test_row)

# conf = SparkConf().set("spark.jars", "/Users/gallup/study/search-ranking/config/spark-tfrecord_2.12-0.3.3_1.15.0.jar")
#
# sc = SparkContext( conf=conf)
sc, spark = CreateSparkContex()
rdd = spark.sparkContext.parallelize(test_rows)

df = spark.createDataFrame(rdd, schema)
df.show()
df.write.mode("overwrite").format("tfrecord").option("recordType", "Example").save(path)
df = spark.read.format("tfrecord").option("recordType", "Example").load(path)
