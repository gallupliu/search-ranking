from pyalink.alink import *
import pandas as pd
# useLocalEnv(1)
# df = pd.DataFrame([
#     [0, "abcde", "aabce"],
#     [1, "aacedw", "aabbed"],
#     [2, "cdefa", "bbcefa"],
#     [3, "bdefh", "ddeac"],
#     [4, "acedm", "aeefbc"]
# ])
# inOp1 = BatchOperator.fromDataframe(df, schemaStr='id long, text1 string, text2 string')
# inOp2 = StreamOperator.fromDataframe(df, schemaStr='id long, text1 string, text2 string')
# batchOp = StringSimilarityPairwiseBatchOp().setSelectedCols(["text1", "text2"]).setMetric("LEVENSHTEIN").setOutputCol("output")
# batchOp.linkFrom(inOp1).print()
# streamOp = StringSimilarityPairwiseStreamOp().setSelectedCols(["text1", "text2"]).setMetric("COSINE").setOutputCol("output")
# streamOp.linkFrom(inOp2).print()
# StreamOperator.execute()
# source = CsvSourceBatchOp()\
# .setFilePath('./review.csv')\
# .setSchemaStr('label long, review string')\
# .setIgnoreFirstLine(True)
#
# source.firstN(10).print()
#
#
# pipeline = Pipeline(
#     Imputer().setSelectedCols(["review"]).setOutputCols(["featureText"]).setStrategy("value").setFillValue("null"),
#     Segment().setSelectedCol("featureText"),
#     StopWordsRemover().setSelectedCol("featureText"),
#     DocCountVectorizer().setFeatureType("TF").setSelectedCol("featureText").setOutputCol("featureVector"),
#     LogisticRegression().setVectorCol("featureVector").setLabelCol("label").setPredictionCol("pred")
# )
#
# model = pipeline.fit(source)
#
# model_path = './lr.model'
# model.save(model_path,overwrite=True)
# model.transform(source).select(["pred", "label", "review"]).firstN(10).print()
#
# # BatchOperator.execute()
#
# model1 = PipelineModel.load(model_path)
#
# model1.transform(source).select(["pred", "label", "review"]).firstN(10).print()


from pyalink.alink import *
import pandas as pd
useLocalEnv(1)
df_data = pd.DataFrame([
    [2, 1, 1],
    [3, 2, 1],
    [4, 3, 2],
    [2, 4, 1],
    [2, 2, 1],
    [4, 3, 2],
    [1, 2, 1],
    [5, 3, 2]
])
input = BatchOperator.fromDataframe(df_data, schemaStr='f0 int, f1 int, label int')
# load data
dataTest = input
colnames = ["f0","f1"]
lr = LogisticRegressionTrainBatchOp().setFeatureCols(colnames).setLabelCol("label")
model = input.link(lr)
filePath = './lr_1.model'
lr.link(AkSinkBatchOp()\
                .setFilePath(FilePath(filePath))\
                .setOverwriteSink(True)\
                .setNumFiles(1))
BatchOperator.execute()
predictor = LogisticRegressionPredictBatchOp().setPredictionCol("pred")
predictor.linkFrom(model, dataTest).print()

lr_model = AkSourceBatchOp().setFilePath(FilePath(filePath))

predictor_1 = LogisticRegressionPredictBatchOp().setPredictionCol("pred")
predictor_1 .linkFrom(lr_model, dataTest).print()





