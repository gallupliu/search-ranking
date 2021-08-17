# -*- coding: utf-8 -*-
# @Time    : 2021/7/10 上午8:18
# @Author  : gallup
# @File    : run_deepfm.py
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from utils.feature_column import SparseFeat,DenseFeat,VarLenSparseFeat,get_item_embed
from models.deepfm import DeepFM



def _parse_function(example_proto):
    item_feats = tf.io.decode_csv(example_proto, record_defaults=DEFAULT_VALUES, field_delim='\t')
    parsed = dict(zip(COL_NAME, item_feats))

    feature_dict = {}
    for feat_col in feature_columns:
        print(parsed)
        if isinstance(feat_col, VarLenSparseFeat):
            if feat_col.weight_name is not None:
                kvpairs = tf.strings.split([parsed[feat_col.name]], ',').values[:feat_col.maxlen]
                kvpairs = tf.strings.split(kvpairs, ':')
                kvpairs = kvpairs.to_tensor()
                feat_ids, feat_vals = tf.split(kvpairs, num_or_size_splits=2, axis=1)
                feat_ids = tf.reshape(feat_ids, shape=[-1])
                feat_vals = tf.reshape(feat_vals, shape=[-1])
                if feat_col.dtype != 'string':
                    feat_ids = tf.strings.to_number(feat_ids, out_type=tf.int32)
                feat_vals = tf.strings.to_number(feat_vals, out_type=tf.float32)
                feature_dict[feat_col.name] = feat_ids
                feature_dict[feat_col.weight_name] = feat_vals
            else:
                feat_ids = tf.strings.split([parsed[feat_col.name]], ',').values[:feat_col.maxlen]
                feat_ids = tf.reshape(feat_ids, shape=[-1])
                if feat_col.dtype != 'string':
                    feat_ids = tf.strings.to_number(feat_ids, out_type=tf.int32)
                feature_dict[feat_col.name] = feat_ids

        elif isinstance(feat_col, SparseFeat):
            feature_dict[feat_col.name] = parsed[feat_col.name]

        elif isinstance(feat_col, DenseFeat):
            if not feat_col.pre_embed:
                feature_dict[feat_col.name] = parsed[feat_col.name]
            elif feat_col.pre_embed == 'char':
                if feat_col.reduce_type is not None:
                    print('pre_embed:{0}'.format(feat_col.pre_embed))
                    keys = tf.strings.split(parsed[feat_col.pre_embed], ' ')
                    emb = tf.nn.embedding_lookup(params=CHAR_EMBEDDING, ids=CHAR_ID2IDX.lookup(keys))
                    emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
                    feature_dict[feat_col.name] = emb
                else:
                    print(feat_col.pre_embed, parsed[feat_col.pre_embed])
                    emb = tf.nn.embedding_lookup(params=CHAR_EMBEDDING,
                                                 ids=CHAR_ID2IDX.lookup(parsed[feat_col.pre_embed]))
                    feature_dict[feat_col.name] = emb
            elif feat_col.reduce_type is not None:
                keys = tf.strings.split(parsed[feat_col.pre_embed], ',')
                emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(keys))
                emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
                feature_dict[feat_col.name] = emb
            else:
                emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(parsed[feat_col.pre_embed]))
                feature_dict[feat_col.name] = emb
        else:
            raise Exception("unknown feature_columns....")

    label = parsed['act']

    return feature_dict, label

if __name__ == "__main__":
    print(tf.__version__)
    ########################################################################
    #################数据预处理##############
    ########################################################################
    # 获取char embedding及其查找关系
    # embedding_dim = 32
    # char_file_names = ['../data/char.json']
    # CHAR_ID2IDX, CHAR_EMBEDDING = get_item_embed(char_file_names,embedding_dim)

    # 筛选实体标签categorical 用于定义映射关系
    DICT_CATEGORICAL = {"topic_id": [str(i) for i in range(0, 700)],
                        "keyword_id": [str(i) for i in range(0, 10)],
                        }

    feature_columns = [
        SparseFeat(name="topic_id", voc_size=700, hash_size=None, share_embed=None, embed_dim=8, dtype='int32'),
        SparseFeat(name="keyword_id", voc_size=10, hash_size=None, share_embed=None, embed_dim=8, dtype='int32'),
        SparseFeat(name='client_type', voc_size=2, hash_size=None, share_embed=None, embed_dim=8, dtype='int32'),
        SparseFeat(name='post_type', voc_size=2, hash_size=None, share_embed=None, embed_dim=8, dtype='int32'),
        VarLenSparseFeat(name="follow_topic_id", voc_size=700, hash_size=None, share_embed='topic_id', weight_name=None,
                         combiner='sum', embed_dim=8, maxlen=20, dtype='int32'),
        VarLenSparseFeat(name="all_topic_fav_7", voc_size=700, hash_size=None, share_embed='topic_id',
                         weight_name='all_topic_fav_7_weight', combiner='sum', embed_dim=8, maxlen=5, dtype='int32'),
    ]

    # 线性侧特征及交叉侧特征buda
    linear_feature_columns_name = ["all_topic_fav_7", "follow_topic_id", 'client_type', 'post_type', "topic_id",
                                   "keyword_id"]
    fm_group_column_name = ["topic_id", "follow_topic_id", "all_topic_fav_7", "keyword_id"]

    linear_feature_columns = [col for col in feature_columns if col.name in linear_feature_columns_name]
    fm_group_columns = [col for col in feature_columns if col.name in fm_group_column_name]

    DEFAULT_VALUES = [[0], [''], [0.0], [0.0], [0.0],
                      [''], [''], [0.0]]
    COL_NAME = ['act', 'client_id', 'client_type', 'post_type', 'topic_id', 'follow_topic_id', 'all_topic_fav_7',
                'keyword_id']

    pad_shapes = {}
    pad_values = {}

    for feat_col in feature_columns:
        if isinstance(feat_col, VarLenSparseFeat):
            max_tokens = feat_col.maxlen
            pad_shapes[feat_col.name] = tf.TensorShape([max_tokens])
            pad_values[feat_col.name] = '0' if feat_col.dtype == 'string' else 0
            if feat_col.weight_name is not None:
                pad_shapes[feat_col.weight_name] = tf.TensorShape([max_tokens])
                pad_values[feat_col.weight_name] = tf.constant(-1, dtype=tf.float32)

        # no need to pad labels
        elif isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                pad_shapes[feat_col.name] = tf.TensorShape([])
                pad_values[feat_col.name] = '0'
            else:
                pad_shapes[feat_col.name] = tf.TensorShape([])
                pad_values[feat_col.name] = 0.0
        elif isinstance(feat_col, DenseFeat):
            if not feat_col.pre_embed:
                pad_shapes[feat_col.name] = tf.TensorShape([])
                pad_values[feat_col.name] = 0.0
            else:
                pad_shapes[feat_col.name] = tf.TensorShape([feat_col.dim])
                pad_values[feat_col.name] = 0.0

    pad_shapes = (pad_shapes, (tf.TensorShape([])))
    pad_values = (pad_values, (tf.constant(0, dtype=tf.int32)))

    filenames = tf.data.Dataset.list_files([
        './user_item_act_test.csv',
    ])
    dataset = filenames.flat_map(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

    batch_size = 2
    dataset = dataset.map(_parse_function, num_parallel_calls=60)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=batch_size)  # 在缓冲区中随机打乱数据
    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=pad_shapes,
                                   padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # 验证集
    filenames_val = tf.data.Dataset.list_files(['./user_item_act_test.csv'])
    dataset_val = filenames_val.flat_map(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

    val_batch_size = 2
    dataset_val = dataset_val.map(_parse_function, num_parallel_calls=60)
    dataset_val = dataset_val.padded_batch(batch_size=val_batch_size,
                                           padded_shapes=pad_shapes,
                                           padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
    dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ########################################################################
    #################模型训练##############
    ########################################################################

    model = DeepFM(linear_feature_columns, fm_group_columns, DICT_CATEGORICAL,dnn_hidden_units=(128, 128), dnn_activation='relu',
                   seed=1024, )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=tf.keras.metrics.AUC(name='auc'))

    log_dir = './tensorboardshare/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
                             histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             write_graph=True,  # 是否存储网络结构图
                             write_images=True,  # 是否可视化参数
                             update_freq='epoch',
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None,
                             profile_batch=20)

    total_train_sample = 100
    total_test_sample = 100
    train_steps_per_epoch = np.floor(total_train_sample / batch_size).astype(np.int32)
    test_steps_per_epoch = np.ceil(total_test_sample / val_batch_size).astype(np.int32)
    history_loss = model.fit(dataset, epochs=3,
                             steps_per_epoch=train_steps_per_epoch,
                             validation_data=dataset_val, validation_steps=test_steps_per_epoch,
                             verbose=1, callbacks=[tbCallBack])
    model_save_path = os.path.join('./', "deepfm/")
    tf.saved_model.save(model, model_save_path)
