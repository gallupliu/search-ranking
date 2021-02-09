from sklearn.preprocessing import LabelEncoder  as LBE
from sklearn.preprocessing import MultiLabelBinarizer as MLB
import pandas as pd
import numpy as np
import json
from pyspark.sql import SparkSession
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
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.feature import NGram
from pyspark.ml.feature import SQLTransformer
import pyspark.sql.types as ArrayType, DoubleType, FloatType


class Feature(object):
    def __init__(self):
        pass

    def load_embedding(self, file_path):
        file = open(file_path, 'r')

        feature_json = json.loads(file)
        return feature_json

    def add_vector(self, df, column, feature_json):
        def parse_vector_from_string(x):
            res = feature_json.get(x)
            return res

        add_embedding = udf(parse_vector_from_string, ArrayType(DoubleType()))
        df = df.withColumn(column + '_vector', add_embedding(column))
        return df

    def calculate_cos(self, df, user_column, item_column):
        def cos_sim(vec1, vec2):
            if (np.linalg.norm(vec1) * np.linalg.norm(vec2)) != 0:
                dot_value = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                return dot_value.tolist()

        cos_sim_udf = udf(cos_sim, FloatType())
        df = df.withColumn(user_column + '_' + item_column + '_cos_dis', cos_sim_udf(user_column, item_column))
        return df
