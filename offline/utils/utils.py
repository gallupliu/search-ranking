import os
import re
import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType, FloatType, StringType


def CreateSparkContex():
    sparkconf = SparkConf().setAppName("MYPRO").set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=sparkconf)
    print("master:" + sc.master)
    sc.setLogLevel("WARN")
    spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()
    return sc, spark


def clean_text(text):
    after_text = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】", "", text)
    return after_text


# 分词
class Segmentation(object):
    def __init__(self, user_dict_path, stop_dict_path):
        self.user_dict_path = user_dict_path
        self.stop_dict_path = stop_dict_path
        jieba.load_userdict(self.user_dict_path)
        jieba.add_word('优思明')
        jieba.del_word('思明')
        # jieba.suggest_freq()
        self.stopwords_list = self.get_stopwords_list(self.stop_dict_path)

    def get_stopwords_list(self, stopwords_path):
        """返回stopwords列表"""
        stopwords_list = [i.strip()
                          for i in codecs.open(stopwords_path).readlines()]
        return stopwords_list

        # 分词

    def cut_sentence(self, sentence):
        """对切割之后的词语进行过滤，去除停用词，保留名词，英文和自定义词库中的词，长度大于2的词"""
        # print(sentence,"*"*100)
        # eg:[pair('今天', 't'), pair('有', 'd'), pair('雾', 'n'), pair('霾', 'g')]
        if not jieba.dt.initialized:
            jieba.load_userdict(self.user_dict_path)
        seg_list = pseg.lcut(sentence.lower())
        # print('before:{}'.format(seg_list))
        # print('stop:{}'.format(self.stopwords_list))
        # for i in seg_list:
        #     print('flag:{}'.format(i.flag))
        seg_list = [i for i in seg_list if i.word not in self.stopwords_list]
        # print('after:{}'.format(seg_list))
        filtered_words_list = []
        for seg in seg_list:
            # print(seg)
            if len(seg.word) <= 1:
                continue
            elif seg.flag == "eng":
                if len(seg.word) <= 2:
                    continue
                else:
                    filtered_words_list.append(seg.word)
            elif seg.flag.startswith("n"):
                filtered_words_list.append(seg.word)
            elif seg.flag in ["x", "eng"]:  # 是自定一个词语或者是英文单词
                filtered_words_list.append(seg.word)
        return filtered_words_list

    def cut(self, df, column):
        clean_udf = udf(clean_text,StringType())
        add_embedding = udf(self.cut_sentence, ArrayType(StringType()))
        df = df.withColumn(column+'_clean',clean_udf(column))
        df = df.withColumn(column+'_list', add_embedding(column+'_clean'))
        return df



if __name__ == "__main__":
    # after_text = clean_text("【镇店之宝】蒙牛 纯牛奶250ml*12盒牛奶整箱【4月新货】")
    after_text = clean_text("双11 双11 避孕套玻尿酸润滑剂安全套琅琊颗粒立体大颗粒型套套成人情趣用品男用")
    print(after_text)
    seg_tool = Segmentation('../data/goods/user_dict.txt', '../data/goods/stopwords.txt')
    print(seg_tool.cut_sentence(after_text))
