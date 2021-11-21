# import six
import os
import pandas as pd
# import numpy as np
# import tensorflow_ranking as tfr
# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#
# tf.compat.v1.enable_eager_execution()
# tf.compat.v1.set_random_seed(1234)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
#
# _TRAIN_DATA_PATH = "data/train-1000-framenet-BERT-context-rel.tfrecords"
# _TEST_DATA_PATH = "data/dev-framenet-BERT-context-rel.tfrecords"
# _VOCAB_PATH = "data/vocab.txt"
# _LIST_SIZE = 250
# _LABEL_FEATURE = "relevance"
# _PADDING_LABEL = -1
# _LEARNING_RATE = 0.05
# _BATCH_SIZE = 32
# _HIDDEN_LAYER_DIMS = ["64", "32", "16"]
# _DROPOUT_RATE = 0.8
# _GROUP_SIZE = 5 # Pointwise scoring.
# _MODEL_DIR = "models/model"
# _NUM_TRAIN_STEPS = 15 * 1000
# _CHECKPOINT_DIR = "chk_points"
# _EMBEDDING_DIMENSION = 20
#
# def context_feature_columns():
#
#     query_column = tf.feature_column.categorical_column_with_vocabulary_file(
#     key="query_tokens",
#     vocabulary_file=_VOCAB_PATH)
#     query_embedding_column = tf.feature_column.embedding_column(
#     query_column, _EMBEDDING_DIMENSION)
#
#     answ_column = tf.feature_column.categorical_column_with_vocabulary_file(
#     key="answer_tokens",
#     vocabulary_file=_VOCAB_PATH)
#     answ_embedding_column = tf.feature_column.embedding_column(
#     answ_column, _EMBEDDING_DIMENSION)
#
#     qid = tf.feature_column.numeric_column(key="qid",dtype=tf.int64)
#
#     overall_bert_encoding_column = tf.feature_column.numeric_column(key="overal_bert_context_encoding_out", shape=768)
#
#     context_features = {"query_tokens": query_embedding_column, "answer_tokens": answ_embedding_column, "qid": qid,
#     "overal_bert_context_encoding_out": overall_bert_encoding_column}
#
#     return context_features
#
# def example_feature_columns():
#     """Returns the example feature columns."""
#
#     expl_column = tf.feature_column.categorical_column_with_vocabulary_file(
#     key="expl_tokens",
#     vocabulary_file=_VOCAB_PATH)
#     expl_embedding_column = tf.feature_column.embedding_column(
#     expl_column, _EMBEDDING_DIMENSION)
#
#     relevance = tf.feature_column.numeric_column(key="relevance",dtype=tf.int64,default_value=_PADDING_LABEL)
#
#     examples_features = {"expl_tokens": expl_embedding_column,"relevance": relevance}
#
#     for fea in range(1,402212):
#         # id, value = fea.split(":")
#         try :
#             feat = tf.feature_column.numeric_column(key=str(fea), dtype=tf.int64,default_value=0)
#             examples_features[""+str(fea)] = feat
#         except :
#             continue
#     return examples_features
#
# def input_fn(path, num_epochs=None):
#     context_feature_spec = tf.feature_column.make_parse_example_spec(
#     context_feature_columns().values())
#     label_column = tf.feature_column.numeric_column(
#     _LABEL_FEATURE, dtype=tf.int64, default_value=_PADDING_LABEL)
#
#     example_feature_spec = tf.feature_column.make_parse_example_spec(
#     list(example_feature_columns().values()) + [label_column])
#     dataset = tfr.data.build_ranking_dataset(
#     file_pattern=path,
#     data_format=tfr.data.EIE,
#     batch_size=_BATCH_SIZE,
#     list_size=_LIST_SIZE,
#     context_feature_spec=context_feature_spec,
#     example_feature_spec=example_feature_spec,
#     reader=tf.data.TFRecordDataset,
#     shuffle=False,
#     num_epochs=num_epochs)
#     features = tf.data.make_one_shot_iterator(dataset).get_next()
#     label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
#     label = tf.cast(label, tf.float32)
#
#     return features, label
#
# #Tranform Input
# def make_transform_fn():
#     def _transform_fn(features, mode):
#         """Defines transform_fn."""
#         example_name = next(six.iterkeys(example_feature_columns()))
#         input_size = tf.shape(input=features[example_name])[1]
#         context_features, example_features = tfr.feature.encode_listwise_features(
#         features=features,
#         input_size=input_size,
#         context_feature_columns=context_feature_columns(),
#         example_feature_columns=example_feature_columns(),
#         mode=mode,
#         scope="transform_layer")
#
#         return context_features, example_features
#     return _transform_fn
#
# # Feature Interactions using scoring_fn
# def make_score_fn():
#     """Returns a scoring function to build EstimatorSpec."""
#
#     def _score_fn(context_features, group_features, mode, params, config):
#         """Defines the network to score a group of documents."""
#         with tf.compat.v1.name_scope("input_layer"):
#         context_input = [
#         tf.compat.v1.layers.flatten(context_features[name])
#         for name in sorted(context_feature_columns())
#         ]
#         group_input = [
#         tf.compat.v1.layers.flatten(group_features[name])
#         for name in sorted(example_feature_columns())
#         ]
#         input_layer = tf.concat(context_input + group_input, 1)
#
#         is_training = (mode == tf.estimator.ModeKeys.TRAIN)
#         cur_layer = input_layer
#         cur_layer = tf.compat.v1.layers.batch_normalization(
#           cur_layer,
#           training=is_training,
#           momentum=0.99)
#
#         for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
#           cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
#           cur_layer = tf.compat.v1.layers.batch_normalization(
#             cur_layer,
#             training=is_training,
#             momentum=0.99)
#           cur_layer = tf.nn.relu(cur_layer)
#           cur_layer = tf.compat.v1.layers.dropout(
#               inputs=cur_layer, rate=_DROPOUT_RATE, training=is_training)
#         logits = tf.compat.v1.layers.dense(cur_layer, units=_GROUP_SIZE)
#         return logits
#     return _score_fn
#
# # Losses, Metrics and Ranking Head
# # Evaluation Metrics
# def eval_metric_fns():
#     metric_fns = {}
#     metric_fns.update({
#     "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
#     tfr.metrics.RankingMetricKey.NDCG, topn=topn)
#     for topn in [1, 3, 5, 10]
#     })
#
#     return metric_fns
#
# _LOSS = tfr.losses.RankingLossKey.APPROX_NDCG_LOSS
# loss_fn = tfr.losses.make_loss_fn(_LOSS)
#
# Ranking Head
# optimizer = tf.compat.v1.train.AdagradOptimizer(
# learning_rate=_LEARNING_RATE)
#
# def _train_op_fn(loss):
#     """Defines train op used in ranking head."""
#
#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     minimize_op = optimizer.minimize(
#     loss=loss, global_step=tf.compat.v1.train.get_global_step())
#     train_op = tf.group([update_ops, minimize_op])
#     return train_op
#
# ranking_head = tfr.head.create_ranking_head(
# loss_fn=loss_fn,
# eval_metric_fns=eval_metric_fns(),
# train_op_fn=_train_op_fn)
#
# # Putting It All Together in a Model Builder
# model_fnc = tfr.model.make_groupwise_ranking_fn(
# group_score_fn=make_score_fn(),
# transform_fn=make_transform_fn(),
# group_size=_GROUP_SIZE,
# ranking_head=ranking_head)
#
# # Train and evaluate the ranker
# def train_and_eval_fn():
#     config_proto = tf.ConfigProto(device_count={'GPU': 3 },log_device_placement=False,
#     allow_soft_placement=False)
#
#     config_proto.gpu_options.per_process_gpu_memory_fraction = 0.8
#     config_proto.gpu_options.allow_growth = True
#
#     run_config = tf.estimator.RunConfig(save_checkpoints_steps=100,
#     # model_dir=_MODEL_DIR,
#     keep_checkpoint_max=5,
#     keep_checkpoint_every_n_hours=5,
#     session_config=config_proto,
#     save_summary_steps=100,
#     log_step_count_steps=100)
#
#     ranker = tf.estimator.Estimator(
#     model_fn=model_fnc,
#     model_dir=_MODEL_DIR,
#     config=run_config)
#
#     train_input_fn = lambda: input_fn(_TRAIN_DATA_PATH)
#     eval_input_fn = lambda: input_fn(_TEST_DATA_PATH, num_epochs=1)
#
#     train_spec = tf.estimator.TrainSpec(
#     input_fn=train_input_fn, max_steps=_NUM_TRAIN_STEPS)
#     eval_spec = tf.estimator.EvalSpec(
#     name="eval",
#     input_fn=eval_input_fn,
#     throttle_secs=15)
#     return (ranker, train_spec, eval_spec)
#
# if name== "main" :
#     ranker, train_spec, eval_spec = train_and_eval_fn()
#     tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)
import tensorflow as tf

from tensorflow.python.feature_column.feature_column import _LazyBuilder

# def test_embedding():
#     # tf.set_random_seed(1)
#     # color_data = {'color': [['R', 'C'], ['G', 'A'], ['B', 'B'], ['A', 'A']]}  # 4行样本
#     # pets = {"pets": ['牛', '奶']}
#     pets = {'pets': [['牛', '奶'], ['液', '体'], ['安', '慕'], ['婴', '儿'], ['口', '味'],["爱",'情']]}
#
#     builder = _LazyBuilder(pets)
#
#
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     fc_path = os.path.join(dir_path, 'char.txt')
#
#     column = tf.feature_column.categorical_column_with_vocabulary_file(
#         key="pets",
#         vocabulary_file=fc_path,
#         num_oov_buckets=0)

    # def truncate_fn():
    #     return tf.slice(serialized_list, [0, 0], [batch_size, list_size])
    #
    # def pad_fn():
    #     return tf.pad(
    #         tensor=serialized_list,
    #         paddings=[[0, 0], [0, list_size - cur_list_size]],
    #         constant_values="")
    #
    # serialized_list = tf.cond(
    #     pred=cur_list_size > list_size, true_fn=truncate_fn, false_fn=pad_fn)
    #
    # color_column_tensor = column._get_sparse_tensors(builder)
    #
    # color_embeding = tf.feature_column.embedding_column(column, 4, combiner='sum')
    # color_embeding_dense_tensor = tf.feature_column.input_layer(pets, [color_embeding])
    #
    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())
    #     session.run(tf.tables_initializer())
    #     print(session.run([color_column_tensor.id_tensor]))
    #     print('embeding' + '_' * 40)
    #     print(session.run([color_embeding_dense_tensor]))

# test_embedding()

def test_data():
    hsy_data = {

        "keyword": ["安 慕 希", "牛 奶", "牛", "奶 粉", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
        "label": [0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    }
    hsy_df = pd.DataFrame(hsy_data)
    print(hsy_df.head(10))
    hsy_df.to_csv('./milk.csv', index=False, sep='\t')


def test_vocab():
    pets = {'pets': [['牛','奶'],['液','体'],['安','慕'],['婴','儿'],['口','味']]}
    # pets = {"pets": ['牛','奶']}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fc_path = os.path.join(dir_path, 'char.txt')

    column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="pets",
        vocabulary_file=fc_path,
        num_oov_buckets=0)

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(pets, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))
        print(session.run([indicator]))

# import requests
# import tensorflow as tf
# import tensorflow_text as tf_text

def input_fn():
    # Create a lookup table for a vocabulary
    _VOCAB = [
        # Special tokens
        b"[UNK]", b"[MASK]", b"[RANDOM]", b"[CLS]", b"[SEP]",
        # Suffixes
        b"##ack", b"##ama", b"##ger", b"##gers", b"##onge", b"##pants", b"##uare",
        b"##vel", b"##ven", b"an", b"A", b"Bar", b"Hates", b"Mar", b"Ob",
        b"Patrick", b"President", b"Sp", b"Sq", b"bob", b"box", b"has", b"highest",
        b"is", b"office", b"the",
    ]

    _START_TOKEN = _VOCAB.index(b"[CLS]")
    _END_TOKEN = _VOCAB.index(b"[SEP]")
    _MASK_TOKEN = _VOCAB.index(b"[MASK]")
    _RANDOM_TOKEN = _VOCAB.index(b"[RANDOM]")
    _UNK_TOKEN = _VOCAB.index(b"[UNK]")
    _MAX_SEQ_LEN = 8
    _MAX_PREDICTIONS_PER_BATCH = 5

    _VOCAB_SIZE = len(_VOCAB)

    lookup_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=_VOCAB,
            key_dtype=tf.string,
            values=tf.range(
                tf.size(_VOCAB, out_type=tf.int64), dtype=tf.int64),
            value_dtype=tf.int64),
        num_oov_buckets=1
    )


    docs = tf.data.Dataset.from_tensor_slices([['an Patrick  Bar has box'], ['an Patrick  Bar has box box'], ["the a box"]])
    tokenizer = tf_text.WhitespaceTokenizer()

    #
    # def truncate_fn():
    #     return tf.slice(serialized_list, [0, 0], [batch_size, list_size])
    #
    # def pad_fn():
    #     return tf.pad(
    #         tensor=serialized_list,
    #         paddings=[[0, 0], [0, list_size - cur_list_size]],
    #         constant_values="")
    #
    # serialized_list = tf.cond(
    #     pred=cur_list_size > list_size, true_fn=truncate_fn, false_fn=pad_fn)

    batch_size = 2
    list_size = 5

    # rt = tf.RaggedTensor.from_row_splits(
    #     values=[[1, 3], [0, 0], [1, 3], [5, 3], [3, 3], [1, 2]],
    #     row_splits=[0, 3, 4, 6])
    # print(rt)
    # print("Shape: {}".format(rt.shape))
    # print("Number of partitioned dimensions: {}".format(rt.ragged_rank))
    # print("Flat values shape: {}".format(rt.flat_values.shape))
    # print("Flat values:\n{}".format(rt.flat_values))

    def _parse(x):
        # tokens_list = tokenizer.tokenize(x)
        rt = tokenizer.tokenize(x)
        tokens_tensor = rt.to_tensor(default_value='', shape=[None, list_size])
        print('tensorfor:{0}'.format(tokens_tensor))
        # Encode tokens
        # tokens_list = tf.ragg(lookup_table.lookup, tokens_tensor)
        tokens_list = lookup_table.lookup(tokens_tensor)
        print(tokens_list,type(tokens_list))

        cur_list_size = tf.shape(input=tokens_list)[1]
        print(cur_list_size)

        # print(tokens_list.flat_values.shape)
        # def truncate_fn():
        #     return tf.slice(tokens_list, [0, 0], [batch_size, list_size])
        #
        # def pad_fn():
        #     return tf.pad(
        #         tensor=tokens_list,
        #         paddings=[[0, 0], [0, list_size - cur_list_size]],
        #         constant_values="")
        #
        # tokens_list = tf.cond(
        #     pred=cur_list_size > list_size, true_fn=truncate_fn, false_fn=pad_fn)

        return tokens_list

    tokenized_docs = docs.map(lambda x: _parse(x))
    iterator = iter(tokenized_docs)
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))

# input_fn()

# test_vocab()
def run_rag_demo():
    ragged_sentences = tf.ragged.constant([
        ['Hi'], ['Welcome', 'to', 'the', 'fair'], ['Have', 'fun']])

    # RaggedTensor -> Tensor
    print(ragged_sentences.to_tensor(default_value='', shape=[None, 10]))

    # Tensor -> RaggedTensor
    x = [[1, 3, -1, -1], [2, -1, -1, -1], [4, 5, 8, 9]]
    print(tf.RaggedTensor.from_tensor(x, padding=-1))
    #RaggedTensor -> SparseTensor
    print(ragged_sentences.to_sparse())

    # SparseTensor -> RaggedTensor
    st = tf.SparseTensor(indices=[[0, 0], [2, 0], [2, 1]],
                         values=['a', 'b', 'c'],
                         dense_shape=[3, 3])
    print(tf.RaggedTensor.from_sparse(st))

# run_rag_demo()

# 用于创建一个特征列
# 并转换一批次数据的一个实用程序方法
# def demo(feature_column):
#   feature_layer = tf.layers.DenseFeatures(feature_column)
#   print(feature_layer(example_batch).numpy())
#
# # 注意到嵌入列的输入是我们之前创建的类别列
# thal_embedding = feature_column.embedding_column(thal, dimension=8)
# demo(thal_embedding)

embedding_initializer = None
# if has_pretrained_embedding:
#   embedding_initializer=tf.contrib.framework.load_embedding_initializer(
#         ckpt_path=xxxx)
# else:
# embedding_initializer=tf.random_uniform_initializer(-1.0, 1.0)
#
# states = tf.feature_column.categorical_column_with_vocabulary_file(
#     key='states', vocabulary_file='/us/states.txt', vocabulary_size=50,
#     num_oov_buckets=5)
#
# embed_column = tf.feature_column.embedding_column(
#     categorical_column=cat_column_with_vocab,
#     dimension=256,   ## this is your pre-trained embedding dimension
#     initializer=embedding_initializer,
#     trainable=False)
#
# price_column = tf.feature_column.numeric_column('price')
#
# columns = [embed_column, price_column]
#
# features = tf.io.parse_example(
#   ..., features=tf.feature_column.make_parse_example_spec(columns + [label_column]))
# labels = features.pop(label_column.name)
#
# features = tf.parse_example(...,
#     features=tf.feature_column.make_parse_example_spec(columns))
# dense_tensor = tf.feature_column.input_layer(features, columns)
# for units in [128, 64, 32]:
#   dense_tensor = tf.layers.dense(dense_tensor, units, tf.nn.relu)
# prediction = tf.layers.dense(dense_tensor, 1)

# def lookup_demo():
#     keys_tensor = tf.constant([b'hello', b'world'])
#     vals_tensor = tf.constant([3, 4], dtype=tf.int64)
#     input_tensor = tf.constant([b'hello', b'ggg'])
#
#     kv_initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
#
#     """ profile_feats.txt
#     hello
#     world
#     """
#     file_initializer = tf.lookup.TextFileInitializer("char.txt",
#                                                      tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
#                                                      tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER)
#
#     table_1 = tf.lookup.StaticHashTable(kv_initializer, -1)
#     table_2 = tf.lookup.StaticHashTable(file_initializer, -1)
#
#     table_3 = tf.lookup.StaticVocabularyTable(kv_initializer, num_oov_buckets=10)
#     table_4 = tf.lookup.StaticVocabularyTable(file_initializer, num_oov_buckets=10)
#
#     out_1 = table_1.lookup(input_tensor)
#     out_2 = table_2.lookup(input_tensor)
#     out_3 = table_3.lookup(input_tensor)
#     out_4 = table_4.lookup(input_tensor)
#
#     with tf.Session() as sess:
#         sess.run(tf.tables_initializer())
#         print(sess.run([out_1, out_2, out_3, out_4]))
#
# lookup_demo()

import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib
#
# def categorical_list_column():
#     column = tf.feature_column.categorical_column_with_vocabulary_list(
#         # 特征列的名称
#         key="feature",
#         # 有效取值列表，列表的下标对应转换的数值。即，value1会被映射为0
#         vocabulary_list=["value1", "value2", "value3"],
#         # 取值的类型，只支持string和integer，这个会根据vocabulary_list自动推断出来
#         dtype=tf.string,
#         # 当取值不在vocabulary_list中时，会被映射的数值，默认为-1
#         # 当该值不为-1时，num_oov_buckets必须设置为0。即两者不能同时起作用
#         default_value=-1,
#         # 作用同default_value，但是两者不能同时起作用。
#         # 将超出的取值映射到[len(vocabulary), len(vocabulary) + num_oov_buckets)内
#         # 默认取值为0
#         # 当该值不为0时，default_value必须设置为-1
#         # 当default_value和num_oov_buckets都取默认值时，会被映射为-1
#         num_oov_buckets=3)
#     feature_cache = feature_column_lib.FeatureTransformationCache(features={
#         # feature对应的值可以为Tensor，也可以为SparseTensor
#         "feature": tf.constant(value=[
#             [["value1", "value2"], ["value3", "value3"]],
#             [["value3", "value5"], ["value4", "value4"]]
#         ])
#     })
#     print(tf.feature_column.embedding_column(column,dimension=10))
#     # IdWeightPair(id_tensor, weight_tensor)
#     return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)
#
# def sequence_categorical_list_column():
#     # 用法同categorical_column_with_vocabulary_list完全一致
#     column = tf.feature_column.sequence_categorical_column_with_vocabulary_list(
#         key="feature",
#         vocabulary_list=["value1", "value2", "value3"],
#         dtype=tf.string,
#         default_value=-1,
#         num_oov_buckets=2)
#     feature_cache = feature_column_lib.FeatureTransformationCache(features={
#         "feature": tf.constant(value=[
#             ["value1", "value2", "value3", "value3"],
#             ["value3", "value5", "value4", "value4"]
#         ])
#     })
#     print(tf.feature_column.embedding_column(column,dimension=10))
#     # IdWeightPair(id_tensor, weight_tensor)
#     return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)
#
# print(categorical_list_column())
# print(sequence_categorical_list_column())

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder

# def test_crossed_column():
#     """ crossed column测试 """
#     featrues = {
#         'price': [['A'], ['B'], ['C']],
#         'color': [['R'], ['G'], ['B']]
#     }
#     price = feature_column.categorical_column_with_vocabulary_list('price', ['A', 'B', 'C', 'D'])
#     color = feature_column.categorical_column_with_vocabulary_list('color', ['R', 'G', 'B'])
#     p_x_c = feature_column.crossed_column([price, color], 16)
#     p_x_c_identy = feature_column.indicator_column(p_x_c)
#     p_x_c_identy_dense_tensor = feature_column.input_layer(featrues, [p_x_c_identy])
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print(session.run([p_x_c_identy_dense_tensor]))
# test_crossed_column()
#
# def test_categorical_column_with_hash_bucket():
#     color_data = {'color': [[2], [5], [-1], [0]]}  # 4行样本
#     builder = _LazyBuilder(color_data)
#     color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
#     color_column_tensor = color_column._get_sparse_tensors(builder)
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print(session.run([color_column_tensor.id_tensor]))
#
#     # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
#     color_column_identy = feature_column.indicator_column(color_column)
#     color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])
#
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print('use input_layer' + '_' * 40)
#         print(session.run([color_dense_tensor]))
#
# test_categorical_column_with_hash_bucket()

# def test_embedding():
#     tf.set_random_seed(1)
#     color_data = {'color': [['R', 'G'], ['G', 'A'], ['B', 'B'], ['A', 'A'],['A','C']]}  # 4行样本
#     builder = _LazyBuilder(color_data)
#     color_column = feature_column.categorical_column_with_vocabulary_list(
#         'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
#     )
#     color_column_tensor = color_column._get_sparse_tensors(builder)
#
#     color_embeding = feature_column.embedding_column(color_column, 5, combiner='sum')
#     color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])
#
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print(session.run([color_column_tensor.id_tensor]))
#         print('embeding' + '_' * 40)
#         print(session.run([color_embeding_dense_tensor]))
#
# test_embedding()
#
# def test_sequence_embedding():
#     tf.set_random_seed(1)
#     color_data = {'color': [['R', 'G'], ['G', 'A'], ['B', 'B'], ['A', 'A'],['A','C']]}  # 4行样本
#     builder = _LazyBuilder(color_data)
#     color_column = feature_column.sequence_categorical_column_with_vocabulary_list(
#         'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
#     )
#     color_column_tensor = color_column._get_sparse_tensors(builder)
#
#     color_embeding = feature_column.embedding_column(color_column, 5, combiner='sum')
#     sequence_feature_layer = tf.keras.experimental.SequenceFeatures(color_embeding)
#     text_input_layer, sequence_length = sequence_feature_layer(color_data)
#     sequence_length_mask = tf.sequence_mask(sequence_length)
#
#     # color_embeding = feature_column.embedding_column(color_column, 4, combiner='sum')
#     # color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])
#
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print(session.run([color_column_tensor.id_tensor]))
#         print('embeding' + '_' * 40)
#         print(session.run([text_input_layer]))
#
# test_sequence_embedding()

# def test_shared_embedding_column_with_hash_bucket():
#     color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]],
#                   'color2': [[2], [5], [-1], [0]]}  # 4行样本
#     builder = _LazyBuilder(color_data)
#     color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
#     color_column_tensor = color_column._get_sparse_tensors(builder)
#     color_column2 = feature_column.categorical_column_with_hash_bucket('color2', 7, dtype=tf.int32)
#     color_column_tensor2 = color_column2._get_sparse_tensors(builder)
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print('not use input_layer' + '_' * 40)
#         print(session.run([color_column_tensor.id_tensor]))
#         print(session.run([color_column_tensor2.id_tensor]))
#
#     # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
#     # color_column_embed1 = feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
#     # print(type(color_column_embed1))
#     # color_dense_tensor1 = feature_column.input_layer(color_data, color_column_embed1)
#
#     color_column_embed = feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
#     print(type(color_column_embed))
#     color_dense_tensor = feature_column.input_layer(color_data, color_column_embed)
#
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print('use input_layer' + '_' * 40)
#         # print(session.run(color_dense_tensor1))
#         print(session.run(color_dense_tensor))
#
# test_shared_embedding_column_with_hash_bucket()

# def test_sequence_shared_embedding_column_with_hash_bucket():
#     color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]],
#                   'color2': [[2], [5], [-1], [0]]}  # 4行样本
#     builder = _LazyBuilder(color_data)
#     color_column = feature_column.sequence_categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
#     color_column_tensor = color_column._get_sparse_tensors(builder)
#     color_column2 = feature_column.sequence_categorical_column_with_hash_bucket('color2', 7, dtype=tf.int32)
#     color_column_tensor2 = color_column2._get_sparse_tensors(builder)
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print('not use input_layer' + '_' * 40)
#         print(session.run([color_column_tensor.id_tensor]))
#         print(session.run([color_column_tensor2.id_tensor]))
#
#     # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
#     # color_column_embed = feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
#     # print(type(color_column_embed))
#     color_embeding = feature_column.shared_embedding_columns([color_column2, color_column], 5, combiner='sum')
#     sequence_feature_layer = tf.keras.experimental.SequenceFeatures(color_embeding)
#     text_input_layer, sequence_length = sequence_feature_layer(color_data)
#     # color_dense_tensor = feature_column.input_layer(color_data, text_input_layer)
#
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print('use input_layer' + '_' * 40)
#         print(session.run(text_input_layer))
#
# test_sequence_shared_embedding_column_with_hash_bucket()
# def test_sequence_keras_shared_embedding_column_with_hash_bucket():
#     color_data = {'query': [[2, 2], [5, 5], [0, -1], [0, 0]],
#                   'item': [[2], [5], [-1], [0]]}  # 4行样本
#     query_input = tf.keras.Input(shape=(1,), name='query', dtype=tf.int64)
#     item_input = tf.keras.Input(shape=(1,), name='item', dtype=tf.int64)
#     video_vocab_list = [0,1,2,3,4,5]
#     dimension=3
#     embed_layer = tf.keras.layers.Embedding(input_dim=len(video_vocab_list), output_dim=dimension)
#     embedded_query_input = embed_layer(query_input)
#     embedded_item_input = embed_layer(item_input)
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print('use input_layer' + '_' * 40)
#         print(session.run([embedded_query_input]))
#         print(session.run([embedded_item_input]))
# def test_categorical_column_with_vocabulary_list():
#     color_data = {'color': [['R', 'R'], ['G', 'R'], ['B', 'G'], ['A', 'A']]}  # 4行样本
#     builder = _LazyBuilder(color_data)
#     color_column = feature_column.categorical_column_with_vocabulary_list(
#         'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
#     )
#
#     color_column_tensor = color_column._get_sparse_tensors(builder)
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print(session.run([color_column_tensor.id_tensor]))
#
#     # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
#     color_column_identy = feature_column.indicator_column(color_column)
#     color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])
#     color_emb_column = tf.feature_column.embedding_column(color_column, dimension=10)
#     color_emb_tensor = feature_column.input_layer(color_data, [color_emb_column])
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print('use input_layer' + '_' * 40)
#         print(session.run([color_dense_tensor]))
#         print(session.run([color_emb_tensor]))
#
# test_categorical_column_with_vocabulary_list()
#
#
# def test_sequence_categorical_column_with_vocabulary_list():
#     color_data = {'color': [['R', 'R'], ['G', 'R'], ['B', 'G'], ['A', 'A']]}  # 4行样本
#     builder = _LazyBuilder(color_data)
#     color_column = feature_column.categorical_column_with_vocabulary_list(
#         'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
#     )
#
#     color_column_tensor = color_column._get_sparse_tensors(builder)
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print(session.run([color_column_tensor.id_tensor]))
#
#     # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
#     color_column_identy = feature_column.indicator_column(color_column)
#     color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])
#     color_emb_column = tf.feature_column.embedding_column(color_column, dimension=10)
#     color_emb_tensor = feature_column.input_layer(color_data, [color_emb_column])
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
#         print('use input_layer' + '_' * 40)
#         print(session.run([color_dense_tensor]))
#         print(session.run([color_emb_tensor]))

# test_sequence_categorical_column_with_vocabulary_list()
#
# def test_embedding_tf1():
#     print(tf.__version__)
#     # 然后将上面的变量存入到ckpt_color(checkpoint文件)中
#     # 在shared_embedding_column中会用到保存的ckpt_color文件
#
#     ckpt_path = ''  # 你指定的ckpt文件
#
#     color_vocab = ['a', 'b', 'c', 'd']
#     color_emb = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
#     color_emb = tf.Variable([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], name='color_emb')
#     # tf.train.saver()
#     color_vocab = ['a', 'b', 'c', 'd']
#     color_data = {'color_clicked': [['a', 'b'], ['c', 'd']],
#                   'color': [['c'], ['a']]}
#
#     color_column = tf.feature_column.categorical_column_with_vocabulary_list('color', vocabulary_list=color_vocab)
#     color_column2 = tf.feature_column.categorical_column_with_vocabulary_list('color_clicked',
#                                                                               vocabulary_list=color_vocab)
#     color_emb_column = tf.feature_column.shared_embedding_columns([color_column, color_column2], 4,
#                                                                   )
#     res = tf.feature_column.input_layer(color_data, color_emb_column)
#
#     # sess = tf.compat.v1.InteractiveSession()
#     # sess.run(tf.compat.v1.global_variables_initializer())
#     #
#     # print(sess.run(res))
#     with tf.Session() as sess:
#         sess.run(tf.compat.v1.global_variables_initializer)
#         sess.run(tf.compat.v1.tables_initializer)
#         sess.run(res)
#
#     # 输出结果和输入对应:
#     # 输入： [['c', ['a', 'b']], ['a', ['c', 'd']]]
#     # 输出： [[3., 4., 1.5, 2.5], [1., 2., 3.5, 4.5]]
#
#
# test_embedding_tf1()