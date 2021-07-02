# import numpy as np
# import pandas as pd
# from pyalink.alink import *
#
# data = np.array([
#     [0, u'二手旧书:医学电磁成像'],
#     [1, u'二手美国文学选读（ 下册 ）李宜燮南开大学出版社 9787310003969'],
#     [2, u'二手正版图解象棋入门/谢恩思主编/华龄出版社'],
#     [3, u'二手中国糖尿病文献索引'],
#     [4, u'二手郁达夫文集（ 国内版 ）全十二册馆藏书']])
#
# test_data = np.array([
#     [0, u'二手旧书:医学电磁成像'],
#     [1, u'伊利纯牛奶 240ml*24']])
# df = pd.DataFrame({"id": data[:, 0], "text": data[:, 1]})
# test_df = pd.DataFrame({"id": test_data[:, 0], "text": test_data[:, 1]})
# inOp1 = BatchOperator.fromDataframe(df, schemaStr='id int, text string')
# inOp2 = StreamOperator.fromDataframe(test_df, schemaStr='id int, text string')
#
# segment = SegmentBatchOp().setSelectedCol("text").linkFrom(inOp1)
# remover = StopWordsRemoverBatchOp().setSelectedCol("text").linkFrom(segment)
# keywords = KeywordsExtractionBatchOp().setSelectedCol("text").setMethod("TF_IDF").setTopN(3).linkFrom(remover)
# keywords.print()
#
# segment = SegmentStreamOp().setSelectedCol("text").linkFrom(inOp2)
# remover = StopWordsRemoverStreamOp().setSelectedCol("text").linkFrom(segment)
# keywords = KeywordsExtractionStreamOp().setSelectedCol("text").setTopN(3).linkFrom(remover)
# keywords.print()
# StreamOperator.execute()


# encoding=utf-8
import jieba
jieba.load_userdict('user.txt')

def get_synonyms(file):
    result = {}
    for line in open(file, "r", encoding='utf-8'):
        seperate_word = line.strip().split("\t")
        print('sep:{0}'.format(seperate_word))

        for i, word in enumerate(seperate_word):
            words = seperate_word.copy()
            print('word:{0} words:{1}'.format(word, words))
            result[word] = ' '.join(words)
    return result


def replace_synonym(string1,synonym_dict):
    # tongyici_tihuan.txt是同义词表，每行是一系列同义词，用tab分割
    # 1读取同义词表：并生成一个字典。


    # 2提升某些词的词频，使其能够被jieba识别出来
    # jieba.suggest_freq("年假", tune=True)

    # 3将语句切分
    seg_list = jieba.cut(string1, cut_all=False)
    f = "/".join(seg_list)  # 不用utf-8编码的话，就不能和tongyici文件里的词对应上
    print(f)

    # 4
    final_sentence = ""
    for word in f.split("/"):
        if word in synonym_dict:
            word = synonym_dict[word]
            final_sentence += word
        else:
            final_sentence += word
    # print final_sentence
    return final_sentence


synonym_dict = get_synonyms("synonym.txt")
print(synonym_dict)
string1 = '年假到底放几天？'
print(replace_synonym(string1,synonym_dict))
