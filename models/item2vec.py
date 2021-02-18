import re
import numpy as np
import multiprocessing
import json
from gensim.models import Word2Vec
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

ALLPUNCS_PATTERN = re.compile(r"[,\_<.>《。》、，：^$#@【】（）]")
NUMBER_PATTERN = re.compile(r"[a-zA-Z0-9]*",re.MULTILINE|re.UNICODE)


def clean_text(text):
    text = re.sub(ALLPUNCS_PATTERN, ' ', text)
    text = re.sub(NUMBER_PATTERN, ' ', text)
    return text


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


def id_data_process(df):
    ids = []
    for seqs in df:
        texts = clean_text(seqs.tolist()[0]).strip().split(' ')
        id = []
        for i in texts :
            if i != '':
                id.append(str(i))
            else:
                continue
        ids.append(id)
    return ids


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

    keys = set()
    item_dict = {}
    for text in texts:
        for char in text:
            keys.add(char)

    with open(save_path + '.json', 'w') as fin:
        for key in keys:
            if key not in item_dict.keys():
                item_dict[key] = model.wv[str(key)].tolist()
        json.dump(item_dict, fin)


if __name__ == '__main__':
    sc, spark = CreateSparkContex()
    char_file_path = '../features/char.csv'
    df = load_data(char_file_path, 'chars')
    char_df = id_data_process(df)
    item2vec(char_df,size=32,window=3,save_path='../data/char')
