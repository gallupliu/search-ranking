# -*- coding: UTF-8 -*-

from pyspark.sql import SparkSession
import os
from sparkxgb import XGBoostClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import  StringIndexer, VectorAssembler

os.environ['JAVA_HOME']="/Library/Java/JavaVirtualMachines/jdk1.8.0_261.jdk/Contents/Home"
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars /Users/gallup/study/search/xgboost-on-pyspark/xgboost_on_pyspark/dependences/xgboost4j-0.90.jar,' \
                                    '/Users/gallup/study/search/xgboost-on-pyspark/xgboost_on_pyspark/dependences/xgboost4j-spark-0.90.jar pyspark-shell'

if __name__=="__main__":

    spark = SparkSession.builder \
        .config("spark.sql.warehouse.dir", "../spark-warehouse") \
        .appName('word_count_app1') \
        .master('local[*]') \
        .getOrCreate()

    df = spark.read.csv("../data/Iris.csv", header=True, inferSchema=True)

    train_df, test_df = df.randomSplit([0.8, 0.2])
    data_col = train_df.drop('IS_TARGET').columns
    assembler = VectorAssembler(inputCols=data_col, outputCol='features')
    stringIndexer = StringIndexer(inputCol='IS_TARGET', outputCol="label")

    xgb=XGBoostClassifier(featuresCol="features", labelCol="label", predictionCol="prediction",
                               numRound=50, colsampleBylevel=0.7, trainTestRatio=0.9,
                               subsample=0.7, seed=123, missing = 0.0, evalMetric="rmse")

    pipeline_model = Pipeline(stages=[assembler, stringIndexer, xgb])

    pipeline_model_fit = pipeline_model.fit(train_df)
    score_df = pipeline_model_fit.transform(test_df)
    score_df.show()





