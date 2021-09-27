# # import warnings
# #
# # warnings.filterwarnings("ignore")
# # import pandas as pd
# # import lightgbm as lgb
# # import numpy as np
# # import xgboost as xgb
# # from catboost import CatBoost, Pool
# # from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, GridSearchCV
# #
# # NEGATIVE_SCALE = 50
# # training_folder = './data/train/'
# # test_folder = './data/test/'
# # csv_folder = './data/csv/'
# #
# # training_file = training_folder + 'features_' + str(NEGATIVE_SCALE) + '.csv'
# # validation_file = training_folder + 'validation_' + str(NEGATIVE_SCALE) + '.csv'
# # training_pd = pd.read_csv(training_file)
# # validation_pd = pd.read_csv(validation_file)
# # # Partial data frames used for quicker gorupby
# # training_pd_part = training_pd[['query_id', 'doc_id']]
# # validation_pd_part = validation_pd[['query_id', 'doc_id']]
# #
# #
# # def get_group(fold_part):
# #     fold_part['group'] = fold_part.groupby('query_id')['query_id'].transform('size')
# #     group = fold_part[['query_id', 'group']].drop_duplicates()['group']
# #     return group
# #
# #
# # def get_eval_group(fold_part):
# #     group = fold_part.groupby('query_id')['query_id'].transform('size')
# #     return group
# #
# #
# # def get_dataset(df):
# #     return df[feat_name], df['relevance']
# #
# #
# # feat_name = []
# # for col in training_pd:
# #     if 'feat_' in col:
# #         feat_name.append(col)
# #
# # for feat in feat_name:
# #     training_pd[feat] = pd.to_numeric(training_pd[feat])
# #     validation_pd[feat] = pd.to_numeric(validation_pd[feat])
# #
# # '''
# #     Read dataset
# # '''
# # train_x, train_y = get_dataset(training_pd)
# # train_group = get_group(training_pd_part)
# # valid_x, valid_y = get_dataset(validation_pd)
# # valid_group = get_group(validation_pd_part)
# # valid_group_full = get_eval_group(validation_pd_part)
# #
# #
# # RECALL_SIZE = 50
# # feature_file = test_folder + f'validation_recalling_{RECALL_SIZE}.csv'
# # feature_pd = pd.read_csv(feature_file)
# # test_pd = feature_pd[['query_id', 'doc_id']]
# # test_pd_group = test_pd.groupby('query_id', sort=False)
# #
# # label_file = csv_folder + 'validation.csv'
# # label_pd = pd.read_csv(label_file)
# #
# # eval_labels = np.zeros((len(label_pd), 10))
# # idx = 0
# # for query in zip(label_pd['query_id'], label_pd['query_label']):
# #     query_id, query_label = query
# #     query_label = [int(i) for i in query_label.split()]
# #     query_label += [np.nan for _ in range(10-len(query_label))]
# #     eval_labels[idx] = np.array(query_label)
# #     idx += 1
# #
#
#
# def get_train_data():
#     pass
#
# def get_analzer():
#     pass
#
# if __name__ == "__main__":
#     #获取训练数据
#
#     #初始化分词器
#
#     #数据清洗等预处理
#
#     #分词
#
#     #获取特征
#
#     #生成特征文件
#
#     #训练
#
#     #测试

import tensorflow as tf

filename = './char.txt'
num_oov_buckets = 3
# input_tensor = tf.constant(["牛", "奶", "液", "口", "味"])
# input_tensor = tf.constant([["牛", "奶"], ["液","体"], ["口", "味"]])
text = tf.constant(["牛  体", "液 体", "牛 奶 口 味"])

input_tensor = tf.strings.split(text)
list_size = 3
batch_size = 3
input_tensor_0 = input_tensor.to_tensor(default_value='<pad>',shape=[3, 3])
input_tensor_1 = input_tensor.to_tensor(default_value='<pad>',shape=[3, 4])
# print(input_tensor)
# print(tf.strings.length(input_tensor).numpy())
# cur_list_size = tf.shape(input=input_tensor)[1]
# #截断
# def truncate_fn():
#     return tf.slice(serialized_list, [0, 0], [batch_size, list_size])
# #补长
# def pad_fn():
#     return tf.pad(
#         tensor=serialized_list,
#         paddings=[[0, 0], [0, list_size - cur_list_size]],
#         constant_values="")
#
# serialized_list = tf.cond(
#     pred=cur_list_size > list_size, true_fn=truncate_fn, false_fn=pad_fn)
# print(serialized_list)

table = tf.lookup.StaticVocabularyTable(
    tf.lookup.TextFileInitializer(
        filename,
        key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        delimiter="\t"),
    num_oov_buckets)
out = table.lookup(input_tensor_0)
print(out)

out_1 = table.lookup(input_tensor_1)
print(out_1)