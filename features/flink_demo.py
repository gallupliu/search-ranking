import numpy as np
import pandas as pd
from pyalink.alink import *

data = np.array([
    [0, u'二手旧书:医学电磁成像'],
    [1, u'二手美国文学选读（ 下册 ）李宜燮南开大学出版社 9787310003969'],
    [2, u'二手正版图解象棋入门/谢恩思主编/华龄出版社'],
    [3, u'二手中国糖尿病文献索引'],
    [4, u'二手郁达夫文集（ 国内版 ）全十二册馆藏书']])
df = pd.DataFrame({"id": data[:, 0], "text": data[:, 1]})
inOp = BatchOperator.fromDataframe(df, schemaStr='id int, text string')

pipeline = (
    Pipeline()
    .add(Segment().setSelectedCol("text"))
    .add(DocCountVectorizer().setSelectedCol("text"))
)

pipeline.fit(inOp).transform(inOp).collectToDataframe()

data = np.array([
    [0, "abcde", "aabce"],
    [1, "aacedw", "aabbed"],
    [2, "cdefa", "bbcefa"],
    [3, "bdefh", "ddeac"],
    [4, "acedm", "aeefbc"]
])
df = pd.DataFrame({"id": data[:, 0], "text1": data[:, 1], "text2": data[:, 2]})
inOp = dataframeToOperator(df, schemaStr='id long, text1 string, text2 string', op_type='batch')

train = StringApproxNearestNeighborTrainBatchOp().setIdCol("id").setSelectedCol("text1").setMetric("SIMHASH_HAMMING_SIM").linkFrom(inOp)
predict = StringApproxNearestNeighborPredictBatchOp().setSelectedCol("text2").setTopN(3).linkFrom(train, inOp)
predict.print()


pipeline = StringNearestNeighbor().setIdCol("id").setSelectedCol("text1").setMetric("LEVENSHTEIN_SIM").setTopN(3)

pipeline.fit(inOp).transform(inOp).print()

data = np.array([
    [0, u'二手旧书:医学电磁成像'],
    [1, u'二手美国文学选读（ 下册 ）李宜燮南开大学出版社 9787310003969'],
    [2, u'二手正版图解象棋入门/谢恩思主编/华龄出版社'],
    [3, u'二手中国糖尿病文献索引'],
    [4, u'二手郁达夫文集（ 国内版 ）全十二册馆藏书']])
df = pd.DataFrame({"id": data[:, 0], "text": data[:, 1]})
inOp1 = BatchOperator.fromDataframe(df, schemaStr='id int, text string')
inOp2 = StreamOperator.fromDataframe(df, schemaStr='id int, text string')

segment = SegmentBatchOp().setSelectedCol("text").linkFrom(inOp1)
remover = StopWordsRemoverBatchOp().setSelectedCol("text").linkFrom(segment)
keywords = KeywordsExtractionBatchOp().setSelectedCol("text").setMethod("TF_IDF").setTopN(3).linkFrom(remover)
keywords.print()

segment = SegmentStreamOp().setSelectedCol("text").linkFrom(inOp2)
remover = StopWordsRemoverStreamOp().setSelectedCol("text").linkFrom(segment)
keywords = KeywordsExtractionStreamOp().setSelectedCol("text").setTopN(3).linkFrom(remover)
keywords.print()
StreamOperator.execute()


data = np.array([
    ["中国 平安 华为"]
])

df = pd.DataFrame({"tokens": data[:, 0]})
inOp1 = dataframeToOperator(df, schemaStr='tokens string', op_type='batch')
inOp2 = dataframeToOperator(df, schemaStr='tokens string', op_type='stream')
train = Word2VecTrainBatchOp().setSelectedCol("tokens").setMinCount(1).setVectorSize(4).linkFrom(inOp1)
predictBatch = Word2VecPredictBatchOp().setSelectedCol("tokens").linkFrom(train, inOp1)

[model,predict] = collectToDataframes(train, predictBatch)
print(model)
print(predict)

predictStream = Word2VecPredictStreamOp(train).setSelectedCol("tokens").linkFrom(inOp2)
predictStream.print(refreshInterval=-1)
StreamOperator.execute()