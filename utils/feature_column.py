import json
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow.keras.layers import *
from collections import namedtuple, OrderedDict
from utils.embedding import EmbeddingLookup, EmbeddingLookupSparse, DenseToSparseTensor, VocabLayer, HashLayer, Add

# 定义参数类型
SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'hash_size', 'share_embed', 'embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed', 'reduce_type', 'dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat',
                              ['name', 'voc_size', 'hash_size', 'share_embed', 'weight_name', 'combiner', 'embed_dim',
                               'maxlen', 'dtype'])


def get_item_embed(file_names, embedding_dim):
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


########################################################################
#################定义输入帮助函数##############
########################################################################

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
def input_from_feature_columns(features, features_columns, embedding_dict, DICT_CATEGORICAL):
    sparse_embedding_list = []
    dense_value_list = []

    for feat_col in features_columns:
        print("input from features feat_col:{}".format(feat_col))
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
                    _input = VocabLayer(keys, mask_value='0')(_input)
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


class FeatureColumn(object):
    def __init__(self):
        self.user_feature_columns = dict()
        self.item_feature_columns = dict()
        self.all_columns = dict()
        self.feature_spec = dict()

    def context_columns(self, features):
        text_columns = {}
        for feat in features:
            column = fc.categorical_column_with_vocabulary_file(
                key=feat, vocabulary_file='./char.txt', vocabulary_size=17,
                num_oov_buckets=5)
            embedding_column = tf.feature_column.embedding_column(
                column, _EMBEDDING_DIMENSION)
            # text_columns.append(embedding_column)
            text_columns[feat] = embedding_column

        # qid = tf.feature_column.numeric_column(key="qid", dtype=tf.int64)
        #
        # overall_bert_encoding_column = tf.feature_column.numeric_column(key="overal_bert_context_encoding_out", shape=768)

        # context_features = {"query_tokens": query_embedding_column, "answer_tokens": answ_embedding_column, "qid": qid,
        #                     "overal_bert_context_encoding_out": overall_bert_encoding_column}

        return text_columns

    def numeric_columns(self, features,shape,):
        text_columns = {}
        for feat in features:
            numeric_column = fc.numeric_column(key=feat, dtype=tf.int64,shape=shape)
            text_columns[feat] = numeric_column

        # qid = tf.feature_column.numeric_column(key="qid", dtype=tf.int64)
        #
        # overall_bert_encoding_column = tf.feature_column.numeric_column(key="overal_bert_context_encoding_out", shape=768)

        # context_features = {"query_tokens": query_embedding_column, "answer_tokens": answ_embedding_column, "qid": qid,
        #                     "overal_bert_context_encoding_out": overall_bert_encoding_column}

        return text_columns

    def categorical_column(self,features):
        pass

    # def create_features_columns(self):
    #     """
    #     features_def["user_classes"] = tf.io.FixedLenFeature([5], tf.int64)  # 用户兴趣
    #     features_def["user_age"] = tf.io.FixedLenFeature([1], tf.int64)  # 用户年龄
    #     features_def["user_gender"] = tf.io.FixedLenFeature([1], tf.int64)  # 用户性别
    #     features_def["user_vector"] = tf.io.FixedLenFeature([128], tf.float32)  # 用户向量
    #     features_def["item_classes"] = tf.io.FixedLenFeature([1], tf.int64)  # item分类
    #     features_def["item_vector"] = tf.io.FixedLenFeature([128], tf.float32)  # item向量
    #     """
    #     user_classes_embed = fc.embedding_column(fc.categorical_column_with_hash_bucket(key="user_classes",
    #                                                                                     hash_bucket_size=40,
    #                                                                                     dtype=tf.int64),
    #                                              dimension=64, combiner='mean',
    #                                              initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1,
    #                                                                                              dtype=tf.float32)
    #                                              )
    #     user_age_embed = fc.embedding_column(fc.categorical_column_with_identity(key="user_age",
    #                                                                              num_bucket=6, dtype=tf.int64),
    #                                          dimension=8, combiner='mean',
    #                                          initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1,
    #                                                                                          dtype=tf.float32)
    #                                          )
    #     user_gender_embed = fc.embedding_column(fc.categorical_column_with_identity(key="user_gender",
    #                                                                                 num_bucket=3, dtype=tf.int64),
    #                                             dimension=8, combiner='mean',
    #                                             initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1,
    #                                                                                             dtype=tf.float32)
    #                                             )
    #
    #     item_classes_embed = fc.embedding_column(fc.categorical_column_with_hash_bucket(key="item_classes",
    #                                                                                     hash_bucket_size=40,
    #                                                                                     dtype=tf.int64),
    #                                              dimension=64, combiner='mean',
    #                                              initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1,
    #                                                                                              dtype=tf.float32)
    #                                              )
    #
    #     user_vector_input = fc.numeric_column(key="user_vector", shape=(128,), default_value=[0.0] * 128,
    #                                           dtype=tf.float32)
    #     item_vector_input = fc.numeric_column(key="item_vector", shape=(128,), default_value=[0.0] * 128,
    #                                           dtype=tf.float32)
    #
    #     for key, value in self.user_feature_columns.items():
    #         self.all_columns[key] = value
    #     for key, value in self.item_feature_columns.items():
    #         self.all_columns[key] = value
    #
    #     self.feature_spec = tf.feature_column.make_parse_example_spec(self.all_columns.values())
    #     return self
    def create_features_columns(self):
        """
        features_def["user_classes"] = tf.io.FixedLenFeature([5], tf.int64)  # 用户兴趣
        features_def["user_age"] = tf.io.FixedLenFeature([1], tf.int64)  # 用户年龄
        features_def["user_gender"] = tf.io.FixedLenFeature([1], tf.int64)  # 用户性别
        features_def["user_vector"] = tf.io.FixedLenFeature([128], tf.float32)  # 用户向量
        features_def["item_classes"] = tf.io.FixedLenFeature([1], tf.int64)  # item分类
        features_def["item_vector"] = tf.io.FixedLenFeature([128], tf.float32)  # item向量
        """
        user_classes_embed = fc.embedding_column(fc.categorical_column_with_hash_bucket(key="user_classes",
                                                                                        hash_bucket_size=40,
                                                                                        dtype=tf.int64),
                                                 dimension=64, combiner='mean',
                                                 initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1,
                                                                                                 dtype=tf.float32)
                                                 )
        user_age_embed = fc.embedding_column(fc.categorical_column_with_identity(key="user_age",
                                                                                 num_bucket=6, dtype=tf.int64),
                                             dimension=8, combiner='mean',
                                             initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1,
                                                                                             dtype=tf.float32)
                                             )
        user_gender_embed = fc.embedding_column(fc.categorical_column_with_identity(key="user_gender",
                                                                                    num_bucket=3, dtype=tf.int64),
                                                dimension=8, combiner='mean',
                                                initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1,
                                                                                                dtype=tf.float32)
                                                )

        item_classes_embed = fc.embedding_column(fc.categorical_column_with_hash_bucket(key="item_classes",
                                                                                        hash_bucket_size=40,
                                                                                        dtype=tf.int64),
                                                 dimension=64, combiner='mean',
                                                 initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1,
                                                                                                 dtype=tf.float32)
                                                 )

        user_vector_input = fc.numeric_column(key="user_vector", shape=(128,), default_value=[0.0] * 128,
                                              dtype=tf.float32)
        item_vector_input = fc.numeric_column(key="item_vector", shape=(128,), default_value=[0.0] * 128,
                                              dtype=tf.float32)

        for key, value in self.user_feature_columns.items():
            self.all_columns[key] = value
        for key, value in self.item_feature_columns.items():
            self.all_columns[key] = value

        self.feature_spec = tf.feature_column.make_parse_example_spec(self.all_columns.values())
        return self
