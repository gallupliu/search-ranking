import numpy as np
import pandas as pd
import json
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

data = {"act": [0, 1, 0, 1, 1],
        "client_id": [144620, 144629, 144620, 144620, 144629],
        "client_type": [0, 1, 0, 0, 1],
        "post_type": [0, 0, 1, 0, 0],
        "topic_id": [1, 1, 1, 177, 1],
        "fellow_topic_id": ["225, 158, 139, 138, 140, 130, 129, 124, 123",
                            "225, 158, 139, 138, 140, 130, 129, 124, 123",
                            "225, 158, 139, 138, 140, 130, 129, 124, 123",
                            "225, 158, 139, 138, 140, 130, 129, 124, 123",
                            "225, 158, 139, 138, 140, 130, 129, 124, 123"],

        "all_topic_fav_7": ["1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
                            "1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
                            "1: 0.4074,177: 0.1217,502: 0.4826"],
        "keyword_id": [1, 2, 3, 4, 4]}
df = pd.DataFrame(data, index=[1, 2, 3, 4, 5])
print(df)
df.to_csv('./user_item_act_test.csv', index=False, sep='\t')

# 'client_id', 'post_id', 'most_reply_topic_name', 'most_post_topic_name', 'follow_topic_id',
#             'all_topic_fav_7', 'all_topic_fav_14', 'topic_id', 'post_type', 'keyword', 'click_seq', 'publisher_id'
recall_data = {"act": [0, 1, 0, 1, 1],
               "client_type": [0, 1, 0, 1, 1],
               "keyword": ["牛", "牛", "牛", "牛", "牛"],
               "client_id": [26697605, 72054397, 92768977, 92858663, 85061501],
               "post_id": [39877457, 39877710, 39878084, 39878084, 39878084],
               "topic_id": [39877457, 39877710, 39878084, 39878084, 39878084],
               "post_type": [0, 1, 0, 1, 1],
               "most_reply_topic_name": [["奶", "牛"], ["粉"], ["牛"], ["中", "国"], ["奶"]],
               "most_post_topic_name": [["奶", "牛"], ["粉"], ["牛"], ["中", "国"], ["奶"]],
               "topic_id": [1, 1, 1, 177, 1],
               "fellow_topic_id": ["177",
                                   "225, 158, 139, 138, 140, 130, 129, 124, 123",
                                   "225, 158, 139, 138, 140, 130, 129, 124, 123",
                                   "225, 158, 139, 138, 140, 130, 129, 124, 123",
                                   "225, 158, 139, 138, 140, 130, 129, 124, 123"],

               "all_topic_fav_7": ["1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
                                   "1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
                                   "1: 0.4074,177: 0.1217,502: 0.4826"],
               "all_topic_fav_14": ["1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
                                    "1: 0.4074,177: 0.1217,502: 0.4826", "1: 0.4074,177: 0.1217,502: 0.4826",
                                    "1: 0.4074,177: 0.1217,502: 0.4826"],
               "click_seq": ["40541953, 40554910, 40555765, 40558999, 40536065",
                             "40541953, 40554910, 40555765, 40558999, 40536065",
                             "40541953, 40554910, 40555765, 40558999, 40536065",
                             "40541953, 40554910, 40555765, 40558999, 40536065",
                             "40541953, 40554910, 40555765, 40558999, 40536065"]}
recall_df = pd.DataFrame(recall_data)
print(recall_df)
recall_df.to_csv('./recall_user_item_act_test.csv', index=False, sep='\t')

ids = []
for seqs in recall_data['click_seq']:
    id = []
    for seq in seqs.split(','):
        id.append(str(seq.replace(' ', '')))
    ids.append(id)
print(ids)
filename = 'item_embed.json'

# 训练skip-gram 模型
model = Word2Vec(ids, size=32, window=10, min_count=1, iter=10, workers=multiprocessing.cpu_count())
# model = Word2Vec(LineSentence(recall_data['click_seq']), size=32, window=3, min_count=1, workers=multiprocessing.cpu_count())
# model.save(outp1)
save_path = './item_model.vector'
model.wv.save_word2vec_format(save_path, binary=False)

item_dict = {}
values = set(ids[0])


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


with open(filename, 'w') as file_obj:
    for value in values:
        item_dict[value] = model.wv[str(value)].tolist()
    json.dump(item_dict, file_obj)
