from collections import namedtuple, OrderedDict
import datetime
import json
import math
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

embedding_dim = 32
# 'act', 'client_id', 'post_id', 'topic_id','post_type','client_type', 'follow_topic_id', 'all_topic_fav_7',
#             'click_seq'
# DEFAULT_VALUES = [[0], [''], [''], [0.0], [''], [''], [''], [''],['']]
# COL_NAME = ['act', 'client_id', 'post_id', 'client_type', 'follow_topic_id', 'all_topic_fav_7', 'topic_id',
#             ]
DEFAULT_VALUES = [[0], [0.0], [''], [''], [''], [''], [''], [''], ['']]
COL_NAME = ['act', 'client_type', "keyword", 'post_id', 'post_type', 'topic_id', 'follow_topic_id', 'all_topic_fav_7',
            'click_seq'
            ]

SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'share_embed', 'embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed', 'reduce_type', 'dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat',
                              ['name', 'voc_size', 'share_embed', 'weight_name', 'embed_dim', 'maxlen', 'dtype'])

feature_columns = [SparseFeat(name="topic_id", voc_size=700, share_embed=None, embed_dim=16, dtype='string'),
                   SparseFeat(name='client_type', voc_size=2, share_embed=None, embed_dim=8, dtype='float32'),
                   VarLenSparseFeat(name="follow_topic_id", voc_size=700, share_embed='topic_id', weight_name=None,
                                    embed_dim=16, maxlen=20, dtype='string'),
                   VarLenSparseFeat(name="all_topic_fav_7", voc_size=700, share_embed='topic_id',
                                    weight_name='all_topic_fav_7_weight', embed_dim=16, maxlen=5, dtype='string'),
                   DenseFeat(name='item_embed', pre_embed='post_id', reduce_type=None, dim=embedding_dim,
                             dtype='float32'),
                   DenseFeat(name='client_embed', pre_embed='post_id', reduce_type='mean', dim=embedding_dim,
                             dtype='float32'),
                   DenseFeat(name='keyword_embed', pre_embed='keyword', reduce_type='mean', dim=embedding_dim,
                             dtype='float32'),
                   ]

# 用户特征及贴子特征
user_feature_columns_name = ["follow_topic_id", 'all_topic_fav_7', 'client_type', 'client_embed',
                             'keyword_embed']
item_feature_columns_name = ["topic_id", "post_id", "post_type", 'item_embed']
user_feature_columns = [col for col in feature_columns if col.name in user_feature_columns_name]
item_feature_columns = [col for col in feature_columns if col.name in item_feature_columns_name]


def get_item_embed(file_names):
    item_bert_embed = []
    item_id = []
    for file in file_names:
        with open(file, 'r') as f:
            for line in f:
                feature_json = json.loads(line)
                # item_bert_embed.append(feature_json['post_id'])
                # item_id.append(feature_json['values'])
                for k, v in feature_json.items():
                    item_bert_embed.append(v)
                    item_id.append(k)

    print(len(item_id))
    item_id2idx = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=item_id,
            values=range(1, len(item_id) + 1),
            key_dtype=tf.string,
            value_dtype=tf.int32),
        default_value=0)
    item_bert_embed = [[0.0] * embedding_dim] + item_bert_embed
    item_embedding = tf.constant(item_bert_embed, dtype=tf.float32)
    return item_id2idx, item_embedding


# 获取item embedding及其查找关系
item_file_names = ['../data/id.json']
ITEM_ID2IDX, ITEM_EMBEDDING = get_item_embed(item_file_names)

# 获取char embedding及其查找关系
char_file_names = ['../data/char.json']
CHAR_ID2IDX, CHAR_EMBEDDING = get_item_embed(char_file_names)

# 定义离散特征集合 ，离散特征vocabulary
DICT_CATEGORICAL = {"topic_id": [str(i) for i in range(0, 700)],
                    "client_type": [0, 1]
                    }


def _parse_function(example_proto):
    item_feats = tf.io.decode_csv(example_proto, record_defaults=DEFAULT_VALUES, field_delim='\t')
    parsed = dict(zip(COL_NAME, item_feats))
    print('paresed:{0}'.format(parsed))
    feature_dict = {}
    for feat_col in feature_columns:
        print('feat_col：{0}'.format(feat_col))
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
            print('name:{0},value:{1}'.format(feat_col.name, parsed[feat_col.name]))
            feature_dict[feat_col.name] = parsed[feat_col.name]

        elif isinstance(feat_col, DenseFeat):
            print('feat_col.pre_embed')
            if feat_col.pre_embed is None:
                feature_dict[feat_col.name] = parsed[feat_col.name]
            elif feat_col.pre_embed == 'post_id':
                if feat_col.reduce_type is not None:
                    print('pre_embed:{0}'.format(feat_col.pre_embed))
                    keys = tf.strings.split(parsed[feat_col.pre_embed], ',')
                    emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(keys))
                    emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
                    feature_dict[feat_col.name] = emb
                else:
                    print(feat_col.pre_embed, parsed[feat_col.pre_embed])
                    emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING,
                                                 ids=ITEM_ID2IDX.lookup(parsed[feat_col.pre_embed]))
                    feature_dict[feat_col.name] = emb
            elif feat_col.pre_embed == 'keyword':
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

        else:
            raise Exception("unknown feature_columns....")

    label = parsed['act']

    return feature_dict, label


pad_shapes = {}
pad_values = {}

for feat_col in feature_columns:
    if isinstance(feat_col, VarLenSparseFeat):
        max_tokens = feat_col.maxlen
        print('max_tokens:{0}'.format(max_tokens))
        pad_shapes[feat_col.name] = tf.TensorShape([max_tokens])
        pad_values[feat_col.name] = ''
        if feat_col.weight_name is not None:
            pad_shapes[feat_col.weight_name] = tf.TensorShape([max_tokens])
            pad_values[feat_col.weight_name] = tf.constant(-1, dtype=tf.float32)

    # no need to pad labels
    elif isinstance(feat_col, SparseFeat):
        if feat_col.dtype == 'string':
            pad_shapes[feat_col.name] = tf.TensorShape([])
            pad_values[feat_col.name] = '9999'
        else:
            pad_shapes[feat_col.name] = tf.TensorShape([])
            pad_values[feat_col.name] = 0.0
    elif isinstance(feat_col, DenseFeat):
        if feat_col.pre_embed is None:
            pad_shapes[feat_col.name] = tf.TensorShape([])
            pad_values[feat_col.name] = 0.0
        else:
            pad_shapes[feat_col.name] = tf.TensorShape([feat_col.dim])
            pad_values[feat_col.name] = 0.0

pad_shapes = (pad_shapes, (tf.TensorShape([])))
pad_values = (pad_values, (tf.constant(0, dtype=tf.int32)))

filenames = tf.data.Dataset.list_files([
    './recall_user_item_act_test.csv'
])
dataset = filenames.flat_map(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

batch_size = 2
dataset = dataset.map(_parse_function, num_parallel_calls=60)
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size=batch_size * 2)  # 在缓冲区中随机打乱数据
dataset = dataset.padded_batch(batch_size=batch_size,
                               padded_shapes=pad_shapes,
                               padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
print('test')
print(next(iter(dataset)))
for example in dataset.take(1):
    print(example)
filenames_val = tf.data.Dataset.list_files(['./recall_user_item_act_test.csv'])
dataset_val = filenames_val.flat_map(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

val_batch_size = 2
dataset_val = dataset_val.map(_parse_function, num_parallel_calls=60)
dataset_val = dataset_val.padded_batch(batch_size=val_batch_size,
                                       padded_shapes=pad_shapes,
                                       padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# 离散多值查找表 转稀疏SparseTensor >> EncodeMultiEmbedding >>tf.nn.embedding_lookup_sparse的sp_ids参数中
class SparseVocabLayer(Layer):
    def __init__(self, keys, **kwargs):
        super(SparseVocabLayer, self).__init__(**kwargs)
        vals = tf.range(1, len(keys) + 1)
        vals = tf.constant(vals, dtype=tf.int32)
        keys = tf.constant(keys)
        # print('keys:{0}'.format(keys))
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals), 0)

    def call(self, inputs):
        input_idx = tf.where(tf.not_equal(inputs, ''))
        input_sparse = tf.SparseTensor(input_idx, tf.gather_nd(inputs, input_idx), tf.shape(inputs, out_type=tf.int64))
        return tf.SparseTensor(indices=input_sparse.indices,
                               values=self.table.lookup(input_sparse.values),
                               dense_shape=input_sparse.dense_shape)


# 自定义Embedding层，初始化时，需要传入预先定义好的embedding矩阵，好处可以共享embedding矩阵
class EncodeMultiEmbedding(Layer):
    def __init__(self, embedding, has_weight=False, **kwargs):

        super(EncodeMultiEmbedding, self).__init__(**kwargs)
        self.has_weight = has_weight
        self.embedding = embedding

    def build(self, input_shape):
        super(EncodeMultiEmbedding, self).build(input_shape)

    def call(self, inputs):
        if self.has_weight:
            idx, val = inputs
            # print('sp_ids:{0}'.format(idx))
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=val, combiner='sum')
        else:

            idx = inputs
            # print('ids:{0}'.format(idx))
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=None, combiner='mean')
        return tf.expand_dims(combiner_embed, 1)

    def get_config(self):
        config = super(EncodeMultiEmbedding, self).get_config()
        config.update({'has_weight': self.has_weight})
        return config


# 稠密权重转稀疏格式输入到tf.nn.embedding_lookup_sparse的sp_weights参数中
class Dense2SparseTensor(Layer):
    def __init__(self):
        super(Dense2SparseTensor, self).__init__()

    def call(self, dense_tensor):
        weight_idx = tf.where(tf.not_equal(dense_tensor, tf.constant(-1, dtype=tf.float32)))
        weight_sparse = tf.SparseTensor(weight_idx, tf.gather_nd(dense_tensor, weight_idx),
                                        tf.shape(dense_tensor, out_type=tf.int64))
        return weight_sparse

    def get_config(self):
        config = super(Dense2SparseTensor, self).get_config()
        return config


# 自定义dnese层含BN， dropout
class CustomDense(Layer):
    def __init__(self, units=32, activation='tanh', dropout_rate=0, use_bn=False, seed=1024, tag_name="dnn", **kwargs):
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        self.tag_name = tag_name

        super(CustomDense, self).__init__(**kwargs)

    # build方法一般定义Layer需要被训练的参数。
    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='random_normal',
                                      trainable=True,
                                      name='kernel_' + self.tag_name)
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='random_normal',
                                    trainable=True,
                                    name='bias_' + self.tag_name)

        if self.use_bn:
            self.bn_layers = tf.keras.layers.BatchNormalization()

        self.dropout_layers = tf.keras.layers.Dropout(self.dropout_rate)
        self.activation_layers = tf.keras.layers.Activation(self.activation, name=self.activation + '_' + self.tag_name)

        super(CustomDense, self).build(input_shape)  # 相当于设置self.built = True

    # call方法一般定义正向传播运算逻辑，__call__方法调用了它。
    def call(self, inputs, training=None, **kwargs):
        fc = tf.matmul(inputs, self.weight) + self.bias
        if self.use_bn:
            fc = self.bn_layers(fc)
        out_fc = self.activation_layers(fc)

        return out_fc

    # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法，保存模型不写这部分会报错
    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({'units': self.units, 'activation': self.activation, 'use_bn': self.use_bn,
                       'dropout_rate': self.dropout_rate, 'seed': self.seed, 'name': self.tag_name})
        return config


# cos 相似度计算层
class Similarity(Layer):

    def __init__(self, gamma=1, axis=-1, type_sim='cos', **kwargs):
        self.gamma = gamma
        self.axis = axis
        self.type_sim = type_sim
        super(Similarity, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Similarity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        query, candidate = inputs
        if self.type_sim == "cos":
            query_norm = tf.norm(query, axis=self.axis)
            candidate_norm = tf.norm(candidate, axis=self.axis)
        cosine_score = tf.reduce_sum(tf.multiply(query, candidate), -1)
        cosine_score = tf.divide(cosine_score, query_norm * candidate_norm + 1e-8)
        cosine_score = tf.clip_by_value(cosine_score, -1, 1.0) * self.gamma
        return tf.expand_dims(cosine_score, 1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'gamma': self.gamma, 'axis': self.axis, 'type': self.type_sim}
        base_config = super(Similarity, self).get_config()
        return base_config.uptate(config)


# 自定损失函数，加权交叉熵损失
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """

    def __init__(self, pos_weight=1.2, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)[:, None]
        ce = ce * (1 - y_true) + self.pos_weight * ce * (y_true)
        #         ce =tf.nn.weighted_cross_entropy_with_logits(
        #             y_true, y_pred, self.pos_weight, name=None
        #         )

        return ce

    def get_config(self, ):
        config = {'pos_weight': self.pos_weight, 'from_logits': self.from_logits, 'name': self.name}
        base_config = super(WeightedBinaryCrossEntropy, self).get_config()
        return base_config.uptate(config)


# 定义model输入特征
def build_input_features(features_columns, prefix=''):
    input_features = OrderedDict()
    for feat_col in features_columns:
        if isinstance(feat_col, DenseFeat):
            if feat_col.pre_embed is None:
                input_features[feat_col.name] = Input([1], name=feat_col.name)
            else:
                input_features[feat_col.name] = Input([feat_col.dim], name=feat_col.name)
        elif isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                input_features[feat_col.name] = Input([None], name=feat_col.name, dtype=feat_col.dtype)
            else:
                input_features[feat_col.name] = Input([1], name=feat_col.name, dtype=feat_col.dtype)
        elif isinstance(feat_col, VarLenSparseFeat):
            input_features[feat_col.name] = Input([None], name=feat_col.name, dtype='string')
            if feat_col.weight_name is not None:
                input_features[feat_col.weight_name] = Input([None], name=feat_col.weight_name, dtype='float32')
        else:
            raise TypeError("Invalid feature column in build_input_features: {}".format(feat_col.name))

    return input_features


# 构造自定义embedding层matrix
def build_embedding_matrix(features_columns):
    embedding_matrix = {}
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            if feat_col.dtype == 'string':
                print('build embedding matrix:{0}'.format(feat_col))
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                vocab_size = feat_col.voc_size
                embed_dim = feat_col.embed_dim
                if vocab_name not in embedding_matrix:
                    embedding_matrix[vocab_name] = tf.Variable(
                        initial_value=tf.random.truncated_normal(shape=(vocab_size, embed_dim), mean=0.0,
                                                                 stddev=0.0, dtype=tf.float32), trainable=True,
                        name=vocab_name + '_embed')
    return embedding_matrix


# 构造自定义 embedding层
def build_embedding_dict(features_columns, embedding_matrix):
    embedding_dict = {}
    for feat_col in features_columns:

        if isinstance(feat_col, SparseFeat):
            print(
                'EncodeMultiEmb name:{0},dtype:{1}'.format('EncodeMultiEmb_' + feat_col.name, feat_col.dtype))
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                embedding_dict[feat_col.name] = EncodeMultiEmbedding(embedding=embedding_matrix[vocab_name],
                                                                     name='EncodeMultiEmb_' + feat_col.name)
        elif isinstance(feat_col, VarLenSparseFeat):
            print(
                'EncodeMultiEmb name:{0},dtype:{1},weight_name:{2}'.format(feat_col.name, feat_col.dtype,
                                                                           feat_col.weight_name))
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            if feat_col.weight_name is not None:
                print('embedding:{0}'.format(embedding_matrix))
                print('vocab_name:{0}'.format(vocab_name))
                embedding_dict[feat_col.name] = EncodeMultiEmbedding(embedding=embedding_matrix[vocab_name],
                                                                     has_weight=True,
                                                                     name='EncodeMultiEmb_' + feat_col.name)
            else:
                print('embedding:{0}'.format(embedding_matrix[vocab_name]))
                print('vocab_name:{0}'.format(vocab_name))
                embedding_dict[feat_col.name] = EncodeMultiEmbedding(embedding=embedding_matrix[vocab_name],
                                                                     name='EncodeMultiEmb_' + feat_col.name)

    return embedding_dict


# dense 与 embedding特征输入
def input_from_feature_columns(features, features_columns, embedding_dict):
    sparse_embedding_list = []
    dense_value_list = []

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            print('sparse or var')
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                keys = DICT_CATEGORICAL[vocab_name]
                _input_sparse = SparseVocabLayer(keys)(features[feat_col.name])

        if isinstance(feat_col, SparseFeat):
            print('sparse')
            if feat_col.dtype == 'string':
                print('string')
                _embed = embedding_dict[feat_col.name](_input_sparse)
            else:
                print('not string')
                _embed = Embedding(feat_col.voc_size + 1, feat_col.embed_dim,
                                   embeddings_regularizer=tf.keras.regularizers.l2(0.5), name='Embed_' + feat_col.name)(
                    features[feat_col.name])
            sparse_embedding_list.append(_embed)
        elif isinstance(feat_col, VarLenSparseFeat):
            if feat_col.weight_name is not None:
                _weight_sparse = Dense2SparseTensor()(features[feat_col.weight_name])
                _embed = embedding_dict[feat_col.name]([_input_sparse, _weight_sparse])

            else:
                _embed = embedding_dict[feat_col.name](_input_sparse)
            sparse_embedding_list.append(_embed)

        elif isinstance(feat_col, DenseFeat):
            dense_value_list.append(features[feat_col.name])

        else:
            raise TypeError("Invalid feature column in input_from_feature_columns: {}".format(feat_col.name))

    return sparse_embedding_list, dense_value_list


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise Exception("dnn_feature_columns can not be empty list")


def DSSM(
        user_feature_columns,
        item_feature_columns,
        user_dnn_hidden_units=(256, 256, 128),
        item_dnn_hidden_units=(256, 256, 128),
        user_dnn_dropout=(0, 0, 0),
        item_dnn_dropout=(0, 0, 0),
        out_dnn_activation='tanh',
        gamma=1.2,
        dnn_use_bn=False,
        seed=1024,
        metric='cos'):
    """
    Instantiates the Deep Structured Semantic Model architecture.
    Args:
        user_feature_columns: A list containing user's features used by the model.
        item_feature_columns: A list containing item's features used by the model.
        user_dnn_hidden_units: tuple,tuple of positive integer , the layer number and units in each layer of user tower
        item_dnn_hidden_units: tuple,tuple of positive integer, the layer number and units in each layer of item tower
        out_dnn_activation: Activation function to use in deep net
        dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
        user_dnn_dropout: tuple of float in [0,1), the probability we will drop out a given user tower DNN coordinate.
        item_dnn_dropout: tuple of float in [0,1), the probability we will drop out a given item tower DNN coordinate.
        seed: integer ,to use as random seed.
        gamma: A useful hyperparameter for Similarity layer
        metric: str, "cos" for  cosine
    return: A TF Keras model instance.
    """
    features_columns = user_feature_columns + item_feature_columns
    # 构建 embedding_dict
    embedding_matrix = build_embedding_matrix(features_columns)
    embedding_dict = build_embedding_dict(features_columns, embedding_matrix)

    # user 特征 处理
    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns, embedding_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    # item 特征 处理
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns, embedding_dict)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

    # user tower
    for i in range(len(user_dnn_hidden_units)):
        if i == len(user_dnn_hidden_units) - 1:
            user_dnn_out = CustomDense(units=user_dnn_hidden_units[i], dropout_rate=user_dnn_dropout[i],
                                       use_bn=dnn_use_bn, activation=out_dnn_activation, name='user_embed_out')(
                user_dnn_input)
            break
        user_dnn_input = CustomDense(units=user_dnn_hidden_units[i], dropout_rate=user_dnn_dropout[i],
                                     use_bn=dnn_use_bn, activation='relu', name='dnn_user_' + str(i))(user_dnn_input)

    # item tower
    for i in range(len(item_dnn_hidden_units)):
        if i == len(item_dnn_hidden_units) - 1:
            item_dnn_out = CustomDense(units=item_dnn_hidden_units[i], dropout_rate=item_dnn_dropout[i],
                                       use_bn=dnn_use_bn, activation=out_dnn_activation, name='item_embed_out')(
                item_dnn_input)
            break
        item_dnn_input = CustomDense(units=item_dnn_hidden_units[i], dropout_rate=item_dnn_dropout[i],
                                     use_bn=dnn_use_bn, activation='relu', name='dnn_item_' + str(i))(item_dnn_input)

    score = Similarity(type_sim=metric, gamma=gamma)([user_dnn_out, item_dnn_out])
    output = tf.keras.layers.Activation("sigmoid", name="dssm_out")(score)
    #    score = Multiply()([user_dnn_out, item_dnn_out])
    #    output = Dense(1, activation="sigmoid",name="dssm_out")(score)

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)
    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)

    return model


model = DSSM(
    user_feature_columns,
    item_feature_columns,
    user_dnn_hidden_units=(256, 256, 128),
    item_dnn_hidden_units=(256, 256, 128),
    user_dnn_dropout=(0, 0, 0),
    item_dnn_dropout=(0, 0, 0),
    out_dnn_activation='tanh',
    gamma=1,
    dnn_use_bn=False,
    seed=1024,
    metric='cos')

initial_learning_rate = 0.01


def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * math.exp(-k * epoch)


model.compile(optimizer='adagrad',
              loss={"dssm_out": WeightedBinaryCrossEntropy(),
                    },
              loss_weights=[1.0, ],
              metrics={"dssm_out": [tf.keras.metrics.AUC(name='auc')]}
              )

log_dir = './logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         write_graph=True,  # 是否存储网络结构图
                         write_images=True,  # 是否可视化参数
                         update_freq='epoch',
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None,
                         profile_batch=40)

#
#
total_train_sample = 10
total_test_sample = 10
train_steps_per_epoch = np.floor(total_train_sample / batch_size).astype(np.int32)
test_steps_per_epoch = np.ceil(total_test_sample / val_batch_size).astype(np.int32)
history_loss = model.fit(dataset, epochs=1,
                         steps_per_epoch=train_steps_per_epoch,
                         validation_data=dataset_val, validation_steps=test_steps_per_epoch,
                         verbose=1,
                         callbacks=[tbCallBack, tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=1)])
model_save_path = os.path.join('./', "dssm/")

# 用户塔 item塔定义
user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
# 保存
tf.keras.models.save_model(user_embedding_model, model_save_path + "/dssmUser/001/")
tf.keras.models.save_model(item_embedding_model, model_save_path + "/dssmItem/001/")
#
# user_query = {'all_topic_fav_7': np.array([['294', '88', '60', '1']]),
#               'all_topic_fav_7_weight': np.array([[0.0897, 0.2464, 0.0928, 0.5711, ]]),
#               'follow_topic_id': np.array([['75', '73', '74', '92', '62', '37', '35', '34', '33', ]),
#               'client_type': np.array([0.]),
#               'client_embed': np.array([[-9.936600e-02, 2.752400e-01, -4.314620e-01, 3.393100e-02,
#                                          -5.263000e-02, -4.490300e-01, -3.641180e-01, -3.545410e-01,
#                                          -2.315470e-01, 4.641480e-01, 3.965120e-01, -1.670170e-01,
#                                          -5.480000e-03, -1.646790e-01, 2.522832e+00, -2.946590e-01,
#                                          ......
#                                          - 1.151946e+00, -4.008270e-01, 1.521650e-01, -3.524520e-01,
#                                          4.836160e-01, -1.190920e-01, 5.792700e-02, -6.148070e-01,
#                                          -7.182930e-01, -1.351920e-01, 2.048980e-01, -1.259220e-01]])}
#
# item_query = {
#     'topic_id': np.array(['1']),
#     'item_embed': np.array([[-9.936600e-02, 2.752400e-01, -4.314620e-01, 3.393100e-02,
#                              -5.263000e-02, -4.490300e-01, -3.641180e-01, -3.545410e-01,
#                              -2.315470e-01, 4.641480e-01, 3.965120e-01, -1.670170e-01,
#                              ......
#                              - 1.151946e+00, -4.008270e-01, 1.521650e-01, -3.524520e-01,
#                              4.836160e-01, -1.190920e-01, 5.792700e-02, -6.148070e-01,
#                              -7.182930e-01, -1.351920e-01, 2.048980e-01, -1.259220e-01]]),
# }

# user_embs = user_embedding_model.predict(user_query)
# item_embs = item_embedding_model.predict(item_query)

# 结果：
# user_embs：
# array([[ 0.80766946,  0.13907856, -0.37779272,  0.53268254, -0.3095821 ,
#          0.2213103 , -0.24618168, -0.7127088 ,  0.4502724 ,  0.4282374 ,
#         -0.36033005,  0.43310016, -0.29158285,  0.8743557 ,  0.5113318 ,
#          0.26994514, -0.35604447,  0.33559784, -0.28052363,  0.38596702,
#          0.5038488 , -0.32811972, -0.5471834 , -0.07594685,  0.7006799 ,
#         -0.24201767,  0.31005877, -0.06173763, -0.28473467,  0.61975694,
# ......
#         -0.714099  , -0.5384026 ,  0.38787717, -0.4263588 ,  0.30690318,
#          0.24047776, -0.01420124,  0.15475503,  0.77783686, -0.43002903,
#          0.52561694,  0.37806144,  0.18955356, -0.37184635,  0.5181224 ,
#         -0.18585253,  0.05573007, -0.38589332, -0.7673693 , -0.25266737,
#          0.51427466,  0.47647673,  0.47982445]], dtype=float32)
# item_embs：
# array([[-6.9417924e-01, -3.9942840e-01,  7.2445291e-01, -5.8977932e-01,
#         -5.8792406e-01,  5.3883100e-01, -7.8469634e-01,  6.8996024e-01,
#         -7.6087400e-02, -4.4855604e-01,  8.4910756e-01, -4.7288817e-01,
#         -9.0812451e-01, -4.0452164e-01,  8.8695991e-01, -7.9177713e-01,
# ......
#         -9.7515762e-01, -5.2411711e-01,  9.2708725e-01, -1.3903661e-01,
#          7.8691095e-01, -8.0726832e-01, -7.3851186e-01,  2.7774110e-01,
#         -4.1870885e-02,  4.7335419e-01,  3.4424815e-01, -5.8394599e-01]],
#       dtype=float32)
