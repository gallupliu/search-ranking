import re
import numpy as np
import multiprocessing
import json
import gensim.models
import Word2Vec
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


def CreateSparkContex():
    sparkconf = SparkConf().setAppName("MYPRO").set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=sparkconf)
    print("master:" + sc.master)
    sc.setLogLevel("WARN")
    spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()
    return sc, spark


def load_data(files_path, column):
    df = spark.read.format('csv') \
        .option('header', 'true') \
        .option('delemiter', ',') \
        .load(files_path)
    return np.array(df.select(column).collect())


def item2vec(texts, size, window, save_path):
    """

    :param texts:
    :param size:
    :param window:
    :param save_path:
    :return:
    """
    # skip-gram
    model = Word2Vec(texts, size=size, window=window, min_count=1, iter=10, workers=multiprocessing.cpu_count())
    model.wv.save_word2vec_format(save_path + '.vec', binary=False)

    keys = []
    item_dict = {}
    for text in texts:
        for char in text:
            keys.append(char)

    with open(save_path + '.json', 'w') as fin:
        for key in keys:
            if key not in item_dict.keys():
                item_dict[key] = model.wv[str[key]].tolist()
        json.dump(item_dict, fin)
