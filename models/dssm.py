import json
import numpy as np
import datetime
from collections import namedtuple, OrderedDict
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

# 定义DSSM输入变量参数
SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'hash_size', 'share_embed', 'embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed', 'reduce_type', 'dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat',
                              ['name', 'voc_size', 'hash_size', 'share_embed', 'weight_name', 'combiner', 'embed_dim',
                               'maxlen', 'dtype'])

feature_columns = [
    SparseFeat(name="topic_id", voc_size=700, hash_size=None, share_embed=None, embed_dim=16, dtype='string'),
    VarLenSparseFeat(name="most_post_topic_name", voc_size=700, hash_size=None, share_embed='topic_id',
                     weight_name=None, combiner='sum', embed_dim=16, maxlen=3, dtype='string'),
    VarLenSparseFeat(name="all_topic_fav_7", voc_size=700, hash_size=None, share_embed='topic_id',
                     weight_name='all_topic_fav_7_weight', combiner='sum', embed_dim=16, maxlen=5, dtype='string'),
    DenseFeat(name='item_embed', pre_embed='post_id', reduce_type=None, dim=768, dtype='float32'),
    DenseFeat(name='client_embed', pre_embed='click_seq', reduce_type='mean', dim=768, dtype='float32'),
    ]

# 用户特征及贴子特征
user_feature_columns_name = ["all_topic_fav_7", "follow_topic_id", 'client_embed', ]
item_feature_columns_name = ['item_embed', "topic_id"]
user_feature_columns = [col for col in feature_columns if col.name in user_feature_columns_name]
item_feature_columns = [col for col in feature_columns if col.name in item_feature_columns_name]


def get_item_embed(file_names):
    item_bert_embed_dict = {}
    item_bert_embed = []
    item_bert_id = []
    for file in file_names:
        with open(file, 'r') as f:
            for line in f:
                feature_json = json.loads(line)
                tid = feature_json['tid']
                embedding = feature_json['features'][0]['layers'][0]['values']
                item_bert_embed_dict[tid] = embedding
    for k, v in item_bert_embed_dict.items():
        item_bert_id.append(k)
        item_bert_embed.append(v)

    item_id2idx = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=item_bert_id,
            values=range(1, len(item_bert_id) + 1),
            key_dtype=tf.string,
            value_dtype=tf.int32),
        default_value=0)
    item_bert_embed = [[0.0] * 768] + item_bert_embed
    item_embedding = tf.constant(item_bert_embed, dtype=tf.float32)

    return item_id2idx, item_embedding

file_names = ''
# user_id = get_client_id(ds)
ITEM_ID2IDX, ITEM_EMBEDDING = get_item_embed(file_names)

# 定义离散特征集合 ，离散特征vocabulary
DICT_CATEGORICAL = {"topic_id": [str(i) for i in range(0, 700)],
                    "client_type": [0, 1]
                    }

DEFAULT_VALUES = [[0], [''], [''], [''], [''],
                  [''], [''], [''], [0.0], [''], [''], ['']]
COL_NAME = ['client_id', 'post_id', 'most_reply_topic_name', 'most_post_topic_name', 'follow_topic_id',
            'all_topic_fav_7', 'all_topic_fav_14', 'topic_id', 'post_type', 'keyword', 'click_seq', 'publisher_id']


def _parse_function(example_proto):
    item_feats = tf.io.decode_csv(example_proto, record_defaults=DEFAULT_VALUES, field_delim='\t')
    parsed = dict(zip(COL_NAME, item_feats))

    feature_dict = {}
    for feat_col in feature_columns:
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

    label = 1

    return feature_dict, label


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

    '/recall_user_item_act_test.csv',
])
dataset = filenames.flat_map(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

batch_size = 512
dataset = dataset.map(_parse_function, num_parallel_calls=60)
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size=batch_size)  # 在缓冲区中随机打乱数据
dataset = dataset.padded_batch(batch_size=batch_size,
                               padded_shapes=pad_shapes,
                               padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 验证集
filenames_val = tf.data.Dataset.list_files(['/recall_user_item_act_test_val.csv', ])
dataset_val = filenames_val.flat_map(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

val_batch_size = 512
dataset_val = dataset_val.map(_parse_function, num_parallel_calls=60)
dataset_val = dataset_val.padded_batch(batch_size=val_batch_size,
                                       padded_shapes=pad_shapes,
                                       padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# 多值查找表稀疏SparseTensor >>  EncodeMultiEmbedding
class VocabLayer(Layer):
    def __init__(self, keys, mask_value=None, **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        self.mask_value = mask_value
        vals = tf.range(2, len(keys) + 2)
        vals = tf.constant(vals, dtype=tf.int32)
        keys = tf.constant(keys)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals), 1)

    def call(self, inputs):
        idx = self.table.lookup(inputs)
        if self.mask_value is not None:
            masks = tf.not_equal(inputs, self.mask_value)
            paddings = tf.ones_like(idx) * (0)  # mask成 0
            idx = tf.where(masks, idx, paddings)
        return idx

    def get_config(self):
        config = super(VocabLayer, self).get_config()
        config.update({'mask_value': self.mask_value, })
        return config


class EmbeddingLookupSparse(Layer):
    def __init__(self, embedding, has_weight=False, combiner='sum', **kwargs):

        super(EmbeddingLookupSparse, self).__init__(**kwargs)
        self.has_weight = has_weight
        self.combiner = combiner
        self.embedding = embedding

    def build(self, input_shape):
        super(EmbeddingLookupSparse, self).build(input_shape)

    def call(self, inputs):
        if self.has_weight:
            idx, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=val,
                                                           combiner=self.combiner)
        else:
            idx = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=None,
                                                           combiner=self.combiner)
        return tf.expand_dims(combiner_embed, 1)

    def get_config(self):
        config = super(EmbeddingLookupSparse, self).get_config()
        config.update({'has_weight': self.has_weight, 'combiner': self.combiner})
        return config


class EmbeddingLookup(Layer):
    def __init__(self, embedding, **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.embedding = embedding

    def build(self, input_shape):
        super(EmbeddingLookup, self).build(input_shape)

    def call(self, inputs):
        idx = inputs
        embed = tf.nn.embedding_lookup(params=self.embedding, ids=idx)
        return embed

    def get_config(self):
        config = super(EmbeddingLookup, self).get_config()
        return config


# 稠密转稀疏
class DenseToSparseTensor(Layer):
    def __init__(self, mask_value=-1, **kwargs):
        super(DenseToSparseTensor, self).__init__()
        self.mask_value = mask_value

    def call(self, dense_tensor):
        idx = tf.where(tf.not_equal(dense_tensor, tf.constant(self.mask_value, dtype=dense_tensor.dtype)))
        sparse_tensor = tf.SparseTensor(idx, tf.gather_nd(dense_tensor, idx), tf.shape(dense_tensor, out_type=tf.int64))
        return sparse_tensor

    def get_config(self):
        config = super(DenseToSparseTensor, self).get_config()
        config.update({'mask_value': self.mask_value})
        return config


class HashLayer(Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(HashLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(HashLayer, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        zero = tf.as_string(tf.zeros([1], dtype='int32'))
        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets, name=None)
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x

    def get_config(self, ):
        config = super(HashLayer, self).get_config()
        config.update({'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, })
        return config


class Add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])
        return tf.keras.layers.add(inputs)

    # cos 相似度计算层


class Similarity(Layer):

    def __init__(self, gamma=20, axis=-1, type_sim='cos', neg=3, **kwargs):
        self.gamma = gamma
        self.axis = axis
        self.type_sim = type_sim
        self.neg = neg
        super(Similarity, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Similarity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        query, candidate = inputs
        bs = tf.shape(query)[0]
        tmp = candidate
        # Negative Sampling
        for i in range(self.neg):
            rand = tf.random.uniform([], minval=0, maxval=bs + i, dtype=tf.dtypes.int32, ) % bs
            candidate = tf.concat([candidate,
                                   tf.slice(tmp, [rand, 0], [bs - rand, -1]),
                                   tf.slice(tmp, [0, 0], [rand, -1])], 0
                                  )
        # 扩充至 candidate 一样的维度
        query = tf.tile(query, [self.neg + 1, 1])

        if self.type_sim == "cos":
            query_norm = tf.norm(query, axis=self.axis)
            candidate_norm = tf.norm(candidate, axis=self.axis)

        # cos_sim_raw = query * candidate / (||query|| * ||candidate||)
        cos_sim_raw = tf.reduce_sum(tf.multiply(query, candidate), -1)
        cos_sim_raw = tf.divide(cos_sim_raw, query_norm * candidate_norm + 1e-8)
        cos_sim_raw = tf.clip_by_value(cos_sim_raw, -1, 1.0)
        # 超参数 gamma 20 论文
        cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [self.neg + 1, -1])) * self.gamma
        # 转化为softmax概率矩阵
        prob = tf.nn.softmax(cos_sim)
        # 只取第一列，即正样本列概率。
        logits = tf.slice(prob, [0, 0], [-1, 1])

        return logits

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'gamma': self.gamma, 'axis': self.axis, 'type': self.type_sim}
        base_config = super(Similarity, self).get_config()
        return base_config.uptate(config)


# 定义model输入特征
def build_input_features(features_columns, prefix=''):
    input_features = OrderedDict()

    for feat_col in features_columns:
        if isinstance(feat_col, DenseFeat):
            input_features[feat_col.name] = Input([feat_col.dim], name=feat_col.name)
        elif isinstance(feat_col, SparseFeat):
            input_features[feat_col.name] = Input([1], name=feat_col.name, dtype=feat_col.dtype)
        elif isinstance(feat_col, VarLenSparseFeat):
            input_features[feat_col.name] = Input([None], name=feat_col.name, dtype=feat_col.dtype)
            if feat_col.weight_name is not None:
                input_features[feat_col.weight_name] = Input([None], name=feat_col.weight_name, dtype='float32')
        else:
            raise TypeError("Invalid feature column in build_input_features: {}".format(feat_col.name))

    return input_features


# 构造 自定义embedding层 matrix
def build_embedding_matrix(features_columns, linear_dim=None):
    embedding_matrix = {}
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            vocab_size = feat_col.voc_size + 2
            embed_dim = feat_col.embed_dim if linear_dim is None else 1
            name_tag = '' if linear_dim is None else '_linear'
            if vocab_name not in embedding_matrix:
                embedding_matrix[vocab_name] = tf.Variable(
                    initial_value=tf.random.truncated_normal(shape=(vocab_size, embed_dim), mean=0.0,
                                                             stddev=0.001, dtype=tf.float32), trainable=True,
                    name=vocab_name + '_embed' + name_tag)
    return embedding_matrix


# 构造 自定义embedding层
def build_embedding_dict(features_columns):
    embedding_dict = {}
    embedding_matrix = build_embedding_matrix(features_columns)

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                            name='emb_lookup_' + feat_col.name)
        elif isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            if feat_col.combiner is not None:
                if feat_col.weight_name is not None:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner, has_weight=True,
                                                                          name='emb_lookup_sparse_' + feat_col.name)
                else:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner,
                                                                          name='emb_lookup_sparse_' + feat_col.name)
            else:
                embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                                name='emb_lookup_' + feat_col.name)

    return embedding_dict


# 构造 自定义embedding层
def build_linear_embedding_dict(features_columns):
    embedding_dict = {}
    embedding_matrix = build_embedding_matrix(features_columns, linear_dim=1)
    name_tag = '_linear'

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                            name='emb_lookup_' + feat_col.name + name_tag)
        elif isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            if feat_col.combiner is not None:
                if feat_col.weight_name is not None:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner, has_weight=True,
                                                                          name='emb_lookup_sparse_' + feat_col.name + name_tag)
                else:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner,
                                                                          name='emb_lookup_sparse_' + feat_col.name + name_tag)
            else:
                embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                                name='emb_lookup_' + feat_col.name + name_tag)

    return embedding_dict


# dense 与 embedding特征输入
def input_from_feature_columns(features, features_columns, embedding_dict):
    sparse_embedding_list = []
    dense_value_list = []

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            _input = features[feat_col.name]
            if feat_col.dtype == 'string':
                if feat_col.hash_size is None:
                    vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                    keys = DICT_CATEGORICAL[vocab_name]
                    _input = VocabLayer(keys)(_input)
                else:
                    _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=False)(_input)

            embed = embedding_dict[feat_col.name](_input)
            sparse_embedding_list.append(embed)
        elif isinstance(feat_col, VarLenSparseFeat):
            _input = features[feat_col.name]
            if feat_col.dtype == 'string':
                if feat_col.hash_size is None:
                    vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                    keys = DICT_CATEGORICAL[vocab_name]
                    _input = VocabLayer(keys, mask_value='0', name='Voc_' + feat_col.name)(_input)
                else:
                    _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=True)(_input)
            if feat_col.combiner is not None:
                input_sparse = DenseToSparseTensor(mask_value=0)(_input)
                if feat_col.weight_name is not None:
                    weight_sparse = DenseToSparseTensor()(features[feat_col.weight_name])
                    embed = embedding_dict[feat_col.name]([input_sparse, weight_sparse])
                else:
                    embed = embedding_dict[feat_col.name](input_sparse)
            else:
                embed = embedding_dict[feat_col.name](_input)

            sparse_embedding_list.append(embed)

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


def get_linear_logit(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        linear_logit = Add()([dense_linear_layer, sparse_linear_layer])
        return linear_logit
    elif len(sparse_embedding_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        return sparse_linear_layer
    elif len(dense_value_list) > 0:
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        return dense_linear_layer
    else:
        raise Exception("linear_feature_columns can not be empty list")


def DSSM(
        user_feature_columns,
        item_feature_columns,
        user_dnn_hidden_units=(256, 256, 128),
        item_dnn_hidden_units=(256, 256, 128),
        user_dnn_dropout=(0, 0, 0),
        item_dnn_dropout=(0, 0, 0),
        out_dnn_activation='tanh',
        gamma=20,
        dnn_use_bn=False,
        seed=1024,
        metric='cos'):
    features_columns = user_feature_columns + item_feature_columns
    # 构建 embedding_dict
    embedding_dict = build_embedding_dict(features_columns)

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
            user_dnn_out = Dense(units=user_dnn_hidden_units[i], activation=out_dnn_activation, name='user_embed_out')(
                user_dnn_input)
            break
        user_dnn_input = Dense(units=user_dnn_hidden_units[i], activation=out_dnn_activation,
                               name='dnn_user_' + str(i))(user_dnn_input)

    # item tower
    for i in range(len(item_dnn_hidden_units)):
        if i == len(item_dnn_hidden_units) - 1:
            item_dnn_out = Dense(units=item_dnn_hidden_units[i], activation=out_dnn_activation, name='item_embed_out')(
                item_dnn_input)
            break
        item_dnn_input = Dense(units=item_dnn_hidden_units[i], activation=out_dnn_activation,
                               name='dnn_item_' + str(i))(item_dnn_input)

    score = Similarity(type_sim=metric, gamma=gamma, name='dssm_out')([user_dnn_out, item_dnn_out])

    output = score

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

model.compile(optimizer='adagrad',
              loss={"dssm_out": "binary_crossentropy",
                    },
              loss_weights=[1.0, ]
              )

log_dir = '/mywork/tensorboardshare/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
total_train_sample = 115930
total_test_sample = 1181
train_steps_per_epoch = np.floor(total_train_sample / batch_size).astype(np.int32)
test_steps_per_epoch = np.ceil(total_test_sample / val_batch_size).astype(np.int32)
history_loss = model.fit(dataset, epochs=1,
                         steps_per_epoch=train_steps_per_epoch,
                         validation_data=dataset_val, validation_steps=test_steps_per_epoch,
                         verbose=1, callbacks=[tbCallBack])

# 用户塔 item塔定义
user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
# 保存
tf.keras.models.save_model(user_embedding_model, "/Recall/DSSM/models/dssmUser/001/")
tf.keras.models.save_model(item_embedding_model, "/Recall/DSSM/models/dssmItem/001/")
#
# user_query = {'all_topic_fav_7': np.array([['294', '88', '60', '1']]),
#               'all_topic_fav_7_weight': np.array([[0.0897, 0.2464, 0.0928, 0.5711, ]]),
#               'follow_topic_id': np.array([['75', '73', '74', '92', '62', '37', '35', '34', '33', ]),
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
#
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
