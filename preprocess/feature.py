from sklearn.preprocessing import LabelEncoder  as LBE
from sklearn.preprocessing import MultiLabelBinarizer as MLB
import pandas as pd
import numpy as np
import json
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.types import IntegerType
# pyspark.sql.functions.col(col) Returns a Column based on the given column name.
from pyspark.sql.functions import col, udf
import pyspark.sql.functions as pyf
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.feature import NGram
from pyspark.ml.feature import SQLTransformer
from pyspark.sql.types import ArrayType, DoubleType, FloatType
from pyspark.ml.linalg import Vectors, DenseVector
import ceja
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import cosine_similarity as cosine
from pyspark.sql.types import StringType


class Feature(object):
    def __init__(self):
        # self.feature_json = self.load_embedding(file_path)
        pass

    def load_embedding(self, file_path):
        default_vector = np.random.uniform(low=-0.1, high=0.1, size=32)

        with open(file_path, 'r') as fin:
            feature_json = json.loads(fin.read())
            if feature_json.get('unk', None) is None:
                feature_json['unk'] = default_vector.tolist()
                fout = open(file_path, 'w')
                json.dump(feature_json, fout)
            return feature_json

    def add_vector(self, df, column, feature_json):
        def parse_vector_from_string(text):
            vecs = []

            if text is None:
                return feature_json.get('奶')
            if isinstance(text, list):
                for word in text:
                    pass

            for char in text:
                # print("x:{0} is in vocabulary".format(char))
                res = feature_json.get(char)
                if res is None:
                    # print("x:{0} is not in vocabulary".format(char))
                    res = feature_json.get('奶')

                vecs.append(res)

            avg_vecs = np.mean(np.array(vecs), axis=0).tolist()
            return avg_vecs

        add_embedding = udf(parse_vector_from_string, ArrayType(DoubleType()))
        df = df.withColumn(column + '_vector', add_embedding(column))
        return df

    def calculate_cos(self, df, user_column, item_column, prefix):
        def cos_sim(vec1, vec2):
            if (np.linalg.norm(vec1) * np.linalg.norm(vec2)) != 0:
                dot_value = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                return dot_value.tolist()

        cos_sim_udf = udf(cos_sim, FloatType())
        df = df.withColumn(user_column + '_' + item_column + '_cos_dis',
                           cos_sim_udf(user_column + prefix, item_column + prefix))
        df = df.drop(user_column + prefix, item_column + prefix)
        return df

    def levenshtein_distance(self, df, user_column, item_column):
        df = df.withColumn(user_column + '_' + item_column + "_levenshtein_distance",
                           ceja.damerau_levenshtein_distance(col(user_column), col(item_column)))
        return df

    def hamming_distance(self, df, user_column, item_column):
        df = df.withColumn(user_column + '_' + item_column + "_hamming_distance",
                           ceja.hamming_distance(col(user_column), col(item_column)))
        return df

    def jaro_similarity(self, df, user_column, item_column):
        df = df.withColumn(user_column + '_' + item_column + "_jaro_similarity",
                           ceja.jaro_similarity(col(user_column), col(item_column)))
        return df

    def jaro_winkler_similarity(self, df, user_column, item_column):
        df = df.withColumn(user_column + '_' + item_column + "_levenshtein_distance",
                           ceja.jaro_winkler_similarity(col(user_column), col(item_column)))
        return df

    def longe_common_substring(self, df, user_column, item_column):

        def lcs(string1, string2):
            answer = ""
            if string1 is None or string2 is None:
                return 0
            len1, len2 = len(string1), len(string2)
            for i in range(len1):
                for j in range(len2):
                    lcs_temp = 0
                    match = ''
                    while ((i + lcs_temp < len1) and (j + lcs_temp < len2) and string1[i + lcs_temp] == string2[
                        j + lcs_temp]):
                        match += string2[j + lcs_temp]
                        lcs_temp += 1
                    if (len(match) > len(answer)):
                        answer = match
            return len(answer)

        lcsUDF = udf(lcs, IntegerType())
        df = df.withColumn(user_column + '_' + item_column + "_lcs",
                           lcsUDF(col(user_column), col(item_column)))
        return df

    def binarizer(self, df, column):
        """
        按指定阈值 二值化Binarizer
        """
        # 对连续值根据阈值threshold二值化
        binarizer = Binarizer(threshold=5.1, inputCol=column, outputCol=column + '_binarized_feature')
        binarizedDataFrame = binarizer.transform(df)
        print('Binarizer output with Threshold = %f' % binarizer.getThreshold())
        return binarizedDataFrame

    def buckert(self, df, column):
        """
        按指定边界 分桶Bucketizer
        """
        splits = [-float('inf'), -0.5, 0.0, 0.5, float('inf')]
        # 按给定边界分桶离散化——按边界分桶
        bucketizer = Bucketizer(splits=splits, inputCol=column, outputCol=column + '_bucketed')  # splits指定分桶边界
        bucketedData = bucketizer.transform(df)
        print('Bucketizer output with %d buckets' % (len(bucketizer.getSplits()) - 1))
        return bucketedData

    def quantile_discretizer(self, df, column, num_buckets):
        """
        按指定等分数 分位数分桶QuantileDiscretizer
        """
        print('QuantileDiscretizerExample')
        df = df.repartition(1)
        # 按分位数分桶离散化——分位数离散化
        if isinstance(column, list):
            output_column = [str(v) + '_quant' for v in numic_columns]
            print(len(column), len(output_column), len(num_buckets))
            discretizer = QuantileDiscretizer(relativeError=0.01, handleInvalid="error",
                                              numBucketsArray=num_buckets, inputCols=column,
                                              outputCols=output_column)
        else:
            discretizer = QuantileDiscretizer(numBuckets=num_buckets, relativeError=0.01, handleInvalid="error",
                                              inputCol=column,
                                              outputCol=column + '_quant')  # numBuckets指定分桶数

        result = discretizer.setHandleInvalid("keep").fit(df).transform(df)
        return result

    def max_abs_scaler(self, df, column):
        """
        按列 特征绝对值归一化MaxAbsScaler
        """
        print('MaxAbsScalerExample')

        # 把“每一列”都缩放到[-1,1]之间——最大绝对值缩放
        scaler = MaxAbsScaler(inputCol=column, outputCol=column + '_maxabs')
        scalerModel = scaler.fit(df)
        scaledData = scalerModel.transform(df)
        return scaledData

    def standard_scaler(self, df, column):
        """
        按列 特征标准化StandardScaler
        """
        print('StandScalerExample')
        # 按特征列减均值除标准差——标准化
        scaler = StandardScaler(inputCol=column, outputCol=column + '_standscale', withStd=False, withMean=True)
        scalerModel = scaler.fit(df)
        scaledData = scalerModel.transform(df)
        return scaledData

    def polynomial_expansion(self, df, column):
        """
        按列 构造多项式特征PolynomialExpansion
        """
        print('PolynomialExpansionExample')
        # 按列交叉构造多项式特征
        # 1 x1 x2
        # 2 x1 x2 x1x2 x1^2 x2^2
        # 3 x1 x2 x1x2 x1^2 x2^2 x1^2x2 x1x2^2 x1^3 x2^3
        polyExpasion = PolynomialExpansion(degree=2, inputCol=column, outputCol=column + '_poly')
        polyDF = polyExpasion.transform(df)
        return polyDF

    def onehot(self, df, column):
        categories = df.select(column).distinct().rdd.flatMap(lambda x: x).collect()
        print(categories)
        categories = [0 if v is None else v for v in categories]
        categories.sort()
        for category in categories:
            function = udf(lambda item: 1 if item == category else 0, IntegerType())
            new_column_name = column + '_' + str(category)
            df = df.withColumn(new_column_name, function(col(column)))
        return df

    def oneHotEncodeColumns(self, df, cols, save_path):
        '''
        :param df: 输入spark.DataFrame
        :param cols: 需要编码的列名
        :return: 编码后的新spark.DataFrame
        '''
        newdf = df
        num = 0
        total = len(cols)
        print("正在onehot特征化...")

        @udf(ArrayType(IntegerType()))
        def toDense(v):
            print(v)
            print(Vectors.dense(v).toArray())
            v = DenseVector(v)

            new_array = list([int(x) for x in v])

            return new_array

        for c in cols:
            num += 1
            # print("{0}/{1} 正在onehot特征:{2}".format(num, total, c))
            onehotEncoderPath = save_path + "/onehot-" + c
            print("{0}/{1} 正在onehot特征:{2}".format(num, total, c))
            stringIndexer = StringIndexer(inputCol=c, outputCol=c + "Index", handleInvalid="keep")
            model = stringIndexer.fit(df)
            indexed = model.transform(df)
            ohe = OneHotEncoder(inputCol=c + "Index", outputCol=c + "-sparse", dropLast=False)
            newdf = ohe.fit(indexed).transform(indexed)
            # newdf = newdf.withColumnRenamed(c + "-onehot", c)
            newdf = newdf.withColumn(c + "-onehot", toDense(c + "-sparse")).drop(c + "-sparse")
            # newdf = newdf.withColumnRenamed(c + "-onehot", c)
            ohe.write().overwrite().save(onehotEncoderPath)
        print("完成onehot特征化!")
        # newdf.withColumn('updatetime', pyf.current_timestamp())
        # newdf.write.mode("overwrite").saveAsTable("mkt_mldb_tmp.TRAIN_dfhotable")  #
        return newdf

    def oneHotDecodeColumns(self, df, cols, save_path):
        '''
        :param df: 输入spark.DataFrame
        :param cols: 需要编码的列名
        :return: 编码后的新spark.DataFrame
        '''
        newdf = df
        num = 0
        total = len(cols)
        print("正在onehot特征化...")
        @udf(ArrayType(IntegerType()))
        def toDense(v):
            print(v)
            print(Vectors.dense(v).toArray())
            v = DenseVector(v)

            new_array = list([int(x) for x in v])

            return new_array
        for c in cols:
            num += 1
            # print("{0}/{1} 正在onehot特征:{2}".format(num, total, c))
            onehotEncoderPath = save_path + "/onehot-" + c
            # 线上预测
            ohe = OneHotEncoder.load(onehotEncoderPath)
            stringIndexer = StringIndexer(inputCol=c, outputCol=c + "Index", handleInvalid="keep")
            model = stringIndexer.fit(df)
            indexed = model.transform(df)
            newdf = ohe.fit(indexed).transform(indexed)
            print('decoder')
            newdf.show()
            # newdf = newdf.withColumnRenamed(c + "-onehot", c)
            # newdf = newdf.withColumnRenamed(c + "-sparse", c)
            newdf.show()
            newdf = newdf.withColumn(c + "-onehot", toDense(c + "-sparse")).drop(c + "-sparse")
            # newdf = newdf.withColumnRenamed(c + "-onehot", c)
        print("完成onehot特征化!")
        return newdf

    def multionehot(self, df, column):
        """
        // Prepare training documents from a list of (id, text, label) tuples.
        val data = spark.createDataFrame(Seq(
          (0L, Seq("A", "B")),
          (1L, Seq("B")),
          (2L, Seq.empty),
          (3L, Seq("D", "E"))
        )).toDF("id", "categories")

        // Get distinct tags array
        val tags = data
          .flatMap(r ⇒ r.getAs[Seq[String]]("categories"))
          .distinct()
          .collect()
          .sortWith(_ < _)

        val cvmData = new CountVectorizerModel(tags)
          .setInputCol("categories")
          .setOutputCol("sparseFeatures")
          .transform(data)

        val asDense = udf((v: Vector) ⇒ v.toDense)

        cvmData
          .withColumn("features", asDense($"sparseFeatures"))
          .select("id", "categories", "features")
          .show()

        :param df:
        :param column:
        :return:
        """
        df.select(column).show()
        categories = list(set(df.select(column).distinct().rdd.flatMap(lambda x: print(x, type(x), '\n')).collect()))
        categories = list(
            set(df.select(column).distinct().rdd.flatMap(lambda x: x[0] if x is not None else None).collect()))
        categories.sort(reverse=False)
        # sorted(categories, key=(lambda x: x[0]))
        print(categories)
        cvm = CountVectorizerModel.from_vocabulary(categories, inputCol=column,
                                                   outputCol=column + "_sparse_vec").transform(df)
        cvm.show()

        @udf(ArrayType(IntegerType()))
        def toDense(v):
            print(v)
            print(Vectors.dense(v).toArray())
            v = DenseVector(v)

            new_array = list([int(x) for x in v])

            return new_array

        result = cvm.withColumn('features_vec', toDense(column + "_sparse_vec"))
        result = result.drop(column + "_sparse_vec")

        return result


def CreateSparkContex():
    sparkconf = SparkConf().setAppName("MYPRO").set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=sparkconf)
    print("master:" + sc.master)
    sc.setLogLevel("WARN")
    spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()
    return sc, spark


if __name__ == "__main__":
    sc, spark = CreateSparkContex()
    feature = Feature()
    data = [
        ("jellyfish", "smellyfish", None, 0.8, 1, "[4,3]", [4, 3], ['牛奶', '奶制品'], "A","A"),
        ("li", "lee", None, 0.5, 1, "[3]", [3], ['牛奶', '奶制品'], "A","A"),
        ("luisa", "bruna", 100, 0.6, 2, None, None, ['牛奶', '奶制品'], "B","A"),
        ("martha", "marhta", 0, 0.3, 3, "[2, 3, 4]", [], ['牛奶', '奶制品'], "C","A"),
        ("口罩", "KN95口罩", 10, 0.9, 3, "[1, 2, 3, 4]", [1, 2, 3, 4], ['牛奶', '奶制品'], "D","E"),
        ("北京", "北京市", 20, 0.8, 5, "[2]", [2], ['牛奶', '奶制品'], "D","E"),
        ("纯牛奶", "牛奶", 50, 0.78, 4, "[1, 2, 3]", [1, 2, 3], ['牛奶', '奶制品'], "E","E"),
        ("安慕希", "牛奶", 20, 0.8, 5, "[1, 2]", [1, 2], ['牛奶', '奶制品'], "E","E"),
        ("奶", "牛", 50, 0.7, 6, "[1, 2, 3]", [1, 2, 3], ['牛奶', '奶制品'], "B","E"),
        ("奶", None, None, 0.7, 6, "[1, 2, 3]", [1, 2, 3], None, "C","E"),
        ("奶", None, 50, 0.7, None, "[1, 2, 3]", [1, 2, 3], None, "E","E"),
    ]
    df = spark.createDataFrame(data, ["word1", "word2", "price", "rate", "category_id", "tag_ids", "ids", "tag_texts",
                                      "category","test"])
    df.show()
    df.printSchema()
    actual_df = df.withColumn("hamming_distance", ceja.hamming_distance(col("word1"), col("word2")))
    actual_df = feature.hamming_distance(df, "word1", "word2")
    print("\nHamming distance")
    actual_df.show()

    actual_df = feature.levenshtein_distance(df, "word1", "word2")
    print("\n Damerau-Levenshtein Distance")
    actual_df.show()

    actual_df = feature.jaro_similarity(df, "word1", "word2")
    print("\n jaro_similarity")
    actual_df.show()

    actual_df = feature.jaro_winkler_similarity(df, "word1", "word2")
    print("\n jaro_winkler_similarity")
    actual_df.show()

    actual_df = feature.longe_common_substring(df, "word1", "word2")
    print("\n lcs")
    actual_df.show()
    word_json = feature.load_embedding('../data/char.json')

    actual_df = feature.add_vector(actual_df, "word1", word_json)
    actual_df.show()

    actual_df = feature.add_vector(actual_df, "word2", word_json)
    actual_df.show()

    actual_df = feature.calculate_cos(actual_df, 'word1', 'word2', '_vector')
    actual_df.show()

    numic_columns = ["price", "rate"]
    for column in numic_columns:
        df = df.withColumn(column, df[column].cast(FloatType()).alias(column))
    #     df = feature.quantile_discretizer(df, column=column, num_buckets=4)
    #     df.show()
    #
    #     # df = feature.binarizer(df, column)
    #     # df.show()
    #
    #     df = feature.buckert(df, column)
    #     df.show()
    #     #
    #     # df = feature.standard_scaler(df, column)
    #     # df.show()

    onehot_numic_columns = [str(v) + '_quant' for v in numic_columns]
    df = feature.quantile_discretizer(df, column=numic_columns, num_buckets=[3, 10])
    ids_columns = ["category_id"]
    # for column in ids_columns + onehot_numic_columns:
    #     df = feature.onehot(df, column)

    # df = feature.oneHotEncodeColumns(df, cols=["category"], save_path='./')
    df = feature.oneHotEncodeColumns(df, cols=["category_id"], save_path='./')
    df = feature.oneHotDecodeColumns(df, cols=["category"], save_path='./')
    df.show()
    df.printSchema()

    print([3] * 5)


    # multi_ids_columns = ["tag_ids", "ids"]
    #
    # for column in multi_ids_columns:
    #     df = feature.multionehot(df, column)
    #     df.show()
    #

    # def array_to_string(my_list):
    #     return '[' + ','.join([str(elem) for elem in my_list]) + ']'
    #
    #
    # array_to_string_udf = udf(array_to_string, StringType())
    #
    # df = df.withColumn('categorystr', array_to_string_udf("category")).drop("category")
    # df = df.withColumn('categoryIdStr', array_to_string_udf("category_id")).drop("category_id")
    # df.show()
    # df.withColumn("features", to_json(struct($"features"))).write.csv(.
    df = df.withColumn("category-onehot", pyf.to_json("category-onehot")).withColumn("category_id-onehot", pyf.to_json("category_id-onehot"))
    df.drop("tag_ids").drop("ids").drop("tag_texts").coalesce(1).write.format("com.databricks.spark.csv").option(
        "header", "true").mode(
        "overwrite").save(
        "feature.csv")

    df = df.toPandas()
    df.to_csv('test.csv',index=False)