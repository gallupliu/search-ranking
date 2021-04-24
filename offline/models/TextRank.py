import os

import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs


# 分词
# def textrank(partition):
#     abspath = "../data/goods/"
#
#     # 结巴加载用户词典
#     userDict_path = os.path.join(abspath, "user_dict.txt")
#     jieba.load_userdict(userDict_path)
#
#     # 停用词文本
#     stopwords_path = os.path.join(abspath, "stopwords.txt")
#
#     def get_stopwords_list():
#         """返回stopwords列表"""
#         stopwords_list = [i.strip()
#                           for i in codecs.open(stopwords_path).readlines()]
#         return stopwords_list
#
#     # 所有的停用词列表
#     stopwords_list = get_stopwords_list()
#
#     class TextRank(jieba.analyse.TextRank):
#         def __init__(self, window=20, word_min_len=2):
#             super(TextRank, self).__init__()
#             self.span = window  # 窗口大小
#             self.word_min_len = word_min_len  # 单词的最小长度
#             # 要保留的词性，根据jieba github ，具体参见https://github.com/baidu/lac
#             self.pos_filt = frozenset(
#                 ('n', 'x', 'eng', 'f', 's', 't', 'nr', 'ns', 'nt', "nw", "nz", "PER", "LOC", "ORG"))
#
#         def pairfilter(self, wp):
#             """过滤条件，返回True或者False"""
#
#             if wp.flag == "eng":
#                 if len(wp.word) <= 2:
#                     return False
#
#             if wp.flag in self.pos_filt and len(wp.word.strip()) >= self.word_min_len \
#                     and wp.word.lower() not in stopwords_list:
#                 return True
#
#     # TextRank过滤窗口大小为5，单词最小为2
#     textrank_model = TextRank(window=3, word_min_len=2)
#     allowPOS = ('n', "x", 'eng', 'nr', 'ns', 'nt', "nw", "nz", "c")
#
#     for row in partition:
#         tags = textrank_model.textrank(row.title, topK=5, withWeight=True, allowPOS=allowPOS, withFlag=False)
#         for tag in tags:
#             yield row.query, row.id, tag[0], tag[1]

# class TextRank(jieba.analyse.TextRank):
#     def __init__(self, stopwords_list, window=20, word_min_len=2):
#         super(TextRank, self).__init__()
#         self.stopwords_list = stopwords_list
#         self.span = window  # 窗口大小
#         self.word_min_len = word_min_len  # 单词的最小长度
#         # 要保留的词性，根据jieba github ，具体参见https://github.com/baidu/lac
#         self.pos_filt = frozenset(
#             ('n', 'x', 'eng', 'f', 's', 't', 'nr', 'ns', 'nt', "nw", "nz", "PER", "LOC", "ORG"))
#         self.allowPOS = ('n', "x", 'eng', 'nr', 'ns', 'nt', "nw", "nz", "c")
#
#     def pairfilter(self, wp):
#         """过滤条件，返回True或者False"""
#
#         if wp.flag == "eng":
#             if len(wp.word) <= 2:
#                 return False
#
#         if wp.flag in self.pos_filt and len(wp.word.strip()) >= self.word_min_len \
#                 and wp.word.lower() not in self.stopwords_list:
#             return True
#
#     def train_model(self,df,column):
#         tags = self.textrank(column, topK=5, withWeight=True, allowPOS=self.allowPOS, withFlag=False)
#         for tag in tags:
#             print(tag)
#             # yield row.query, row.id, tag[0], tag[1]
def textrank(stopwords_list):
    class TextRank(jieba.analyse.TextRank):
        def __init__(self, window=20, word_min_len=2):
            super(TextRank, self).__init__()
            self.span = window  # 窗口大小
            self.word_min_len = word_min_len  # 单词的最小长度
            # 要保留的词性，根据jieba github ，具体参见https://github.com/baidu/lac
            self.pos_filt = frozenset(
                ('n', 'x', 'eng', 'f', 's', 't', 'nr', 'ns', 'nt', "nw", "nz", "PER", "LOC", "ORG"))

        def pairfilter(self, wp):
            """过滤条件，返回True或者False"""

            if wp.flag == "eng":
                if len(wp.word) <= 2:
                    return False

            if wp.flag in self.pos_filt and len(wp.word.strip()) >= self.word_min_len \
                    and wp.word.lower() not in stopwords_list:
                return True

    def my_textrank(partition):

        # TextRank过滤窗口大小为5，单词最小为2
        textrank_model = TextRank(window=3, word_min_len=2)
        allowPOS = ('n', "x", 'eng', 'nr', 'ns', 'nt', "nw", "nz", "c")

        for row in partition:
            tags = textrank_model.textrank(row.title, topK=5, withWeight=True, allowPOS=allowPOS, withFlag=False)
            for tag in tags:
                yield row.query, row.id, tag[0], tag[1]
    return my_textrank


# if __name__ == "__main__":
