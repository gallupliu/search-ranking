from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import CountVectorizerModel
from pyspark.ml.feature import IDF
from pyspark.ml.feature import IDFModel


class TFIDF(object):
    def __init__(self, df, column, base_path,top_k=5):
        self.df = df
        self.column = column
        self.base_path = base_path
        self.cv_model = None
        self.idf_model = None
        self.top_k = top_k

    def train_model(self):
        column = self.column
        print('tdidf train model')
        TOPK = self.top_k
        # 总词汇的大小，文本中必须出现的次数
        cv = CountVectorizer(inputCol=self.column, outputCol=self.column + "countFeatures", vocabSize=200 * 10000,
                             minDF=1.0)
        # 训练词频统计模型
        cv_model = cv.fit(self.df)
        cv_model.write().overwrite().save(self.base_path + "./CV.model")
        self.cv_model = self.load_cv_model(self.base_path + "./CV.model")
        # cv_model = CountVectorizerModel.load(self.base_path + "./CV.model")
        # 得出词频向量结果
        cv_result = self.cv_model.transform(self.df)

        # 训练IDF模型
        idf = IDF(inputCol=self.column + "countFeatures", outputCol=self.column + "idfFeatures")
        idfModel = idf.fit(cv_result)
        idfModel.write().overwrite().save(self.base_path + "./IDF.model")

        # idf_model = IDFModel.load(self.base_path + "./IDF.model")
        self.idf_model = self.load_idf_model(self.base_path + "./IDF.model")
        cv_result = cv_model.transform(self.df)
        tfidf_result = self.idf_model.transform(cv_result)
        tfidf_result.show()

        def func(partition):
            print('TFIDF TRAIN MODEL FUNC')
            for row in partition:
                # 找到索引与IDF值并进行排序
                print('row:{0}'.format(row))
                # print(column)
                _ = list(zip(row[column + 'idfFeatures'].indices, row[column + 'idfFeatures'].values))
                # _ = list(zip(row['title_listidfFeatures'].indices, row['title_listidfFeatures'].values))
                _ = sorted(_, key=lambda x: x[1], reverse=True)
                result = _[:TOPK]
                for word_index, tfidf in result:
                    # print(word_index, tfidf)
                    yield row.query, row.id, int(word_index), round(float(tfidf), 4)

        tfidf_result = tfidf_result.rdd.mapPartitions(func).toDF(["query","id","index", "tfidf"])

        return tfidf_result

    def load_cv_model(self, model_path):
        cv_model = CountVectorizerModel.load(model_path)
        return cv_model

    def load_idf_model(self, model_path):
        idf_model = IDFModel.load(model_path)
        return idf_model

    def predict_cv(self):
        # 得出词频向量结果
        cv_result = self.cv_model.transform(self.df)

    def idf(self):
        pass

    def tfidf(self):
        pass
