# import warnings
#
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')  # 忽略警告
# import sys
#
# import gensim
#
# model = gensim.models.Word2Vec.load('./data/model/Word2Vec/wiki.zh.text.model')
# # 与足球最相似的
# word = model.most_similar("足球")
# for each in word:
#     print(each[0], each[1])
#
# print('*' * 20)
#
# word = model.most_similar(positive=['皇上', '国王'], negative=['皇后'])
# for t in word:
#     print(t[0], t[1])
#
# print(model.doesnt_match('太后 妃子 贵人 贵妃 才人'.split(' ')))
# print('*' * 20)
#
# print(model.similarity('书籍', '书本'))
# print('*' * 20)
# print(model.similarity('逛街', '书本'))
#
# import pandas as pd
#
# df1 = pd.DataFrame({'key1':['a','b','c','d'],'key2':['e','f','g','h'],'key3':['i','j','k','l']},index=['k','l','m','n',])
# df2 = pd.DataFrame({'key1':['a','B','c','d','d'],'key2':['e','f','g','h','h'],'key4':['i','j','K','L','H']},index = ['p','q','u','v','z'])
# print(df1)
# print(df2)
# # print(pd.merge(df1,df2,on='key1'))
# # print(pd.merge(df1,df2,on='key2'))
# # print(pd.merge(df1,df2,on=['key1','key2']))
# # print('default')
# # print(pd.merge(df1,df2,on=['key1','key2']))  #可以看到不加on参数，系统自动以个数最多的相同column为参考
# # print('inner')
# # print(pd.merge(df1,df2,on=['key1','key2'],how='inner'))
# print('left')
# print(pd.merge(df1,df2,on=['key1','key2'],how='left'))
# print('right')
# print(pd.merge(df1,df2,on=['key1','key2'],how='right'))
# print('outer')
# print(pd.merge(df1,df2,on=['key1','key2'],how='outer'))
#   key1 key2 key3
# k    a    e    i
# l    b    f    j
# m    c    g    k
# n    d    h    l
#   key1 key2 key4
# p    a    e    i
# q    B    f    j
# u    c    g    K
# v    d    H    L
#   key1 key2_x key3 key2_y key4
# 0    a      e    i      e    i
# 1    c      g    k      g    K
# 2    d      h    l      H    L
#   key1_x key2 key3 key1_y key4
# 0      a    e    i      a    i
# 1      b    f    j      B    j
# 2      c    g    k      c    K
#   key1 key2 key3 key4
# 0    a    e    i    i
# 1    c    g    k    K
#   key1 key2 key3 key4
# 0    a    e    i    i
# 1    c    g    k    K

from joblib import Memory
from sklearn.datasets import load_svmlight_file
mem = Memory("./mycache")

@mem.cache
def get_data():
    data = load_svmlight_file("/Users/gallup/study/ranking/tensorflow_ranking/examples/data/train.txt")
    return data[0], data[1]

X, y = get_data()
print(X,y)