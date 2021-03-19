
triazines = spark.read.format("libsvm")\
    .load("wasbs://publicwasb@mmlspark.blob.core.windows.net/triazines.scale.svmlight")

# print some basic info
print("records read: " + str(triazines.count()))
print("Schema: ")
triazines.printSchema()
triazines.limit(10).toPandas()


train, test = triazines.randomSplit([0.85, 0.15], seed=1)

from mmlspark.vw import VowpalWabbitRegressor
model = (VowpalWabbitRegressor(numPasses=20, args="--holdout_off --loss_function quantile -q :: -l 0.1")
            .fit(train))

scoredData = model.transform(test)
scoredData.limit(10).toPandas()

from mmlspark.train import ComputeModelStatistics
metrics = ComputeModelStatistics(evaluationMetric='regression',
                                 labelCol='label',
                                 scoresCol='prediction') \
            .transform(scoredData)
metrics.toPandas()





