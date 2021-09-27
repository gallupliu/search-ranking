import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from utils.feature_column import get_item_embed
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from deepctr.estimator import DeepFMEstimator
from deepctr.estimator.inputs import input_fn_tfrecord

def context_feature_columns(features):
    text_columns = {}
    for feat in features:
        column = tf.feature_column.categorical_column_with_vocabulary_file(
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


def _parse(serial_exmp, feature_spec):
    item_feats = tf.io.decode_csv(serial_exmp, record_defaults=DEFAULT_VALUES, field_delim='\t')
    spec = tf.feature_column.make_parse_example_spec(feature_spec)

    print('item_feats:{0} {1}'.format(item_feats,type(item_feats)))
    spec.update({"label":tf.io.FixedLenFeature([], tf.float32)})
    # spec["label"] = tf.io.FixedLenFeature([], tf.int32)
    print(spec)
    feats = tf.io.parse_example(item_feats, features=spec)
    labels = feats.pop('label')
    return feats, labels


def train_input_fn(filenames, feature_spec, batch_size, shuffle_buffer_size):
    # Extract lines from input files using the Dataset API.
    filenames = tf.data.Dataset.list_files([
        filenames,
    ])
    dataset = filenames.flat_map(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

    def _input():
        label_column = tf.feature_column.numeric_column(
            'label', dtype=tf.int64, default_value=0)
        feature_spec = tf.feature_column.make_parse_example_spec(
            feature_columns+[label_column] )
        feats = tf.io.parse_example(item_feats, features=spec)

    # Shuffle, repeat, and batch the examples.
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(lambda x: _parse(x, feature_spec), num_parallel_calls=8)
    dataset = dataset.repeat().batch(batch_size).prefetch(1)
    # print(dataset.output_types)
    # print(dataset.output_shapes)
    return dataset


def input_fn_new(data_path, shuffle, num_epochs, batch_size):
    """Generate an input function for the Estimator."""
    vocab = tf.contrib.lookup.index_table_from_file('./char.txt', num_oov_buckets=1)

    def _parse(example):
        num_oov_buckets = 1
        columns = tf.io.decode_csv(example, record_defaults=DEFAULT_VALUES, field_delim='\t')
        parsed = dict(zip(COL_NAME, columns))
        labels = parsed.pop['label']
        feature_dict = {}

        for feat in text_features:
            tokens = parsed[feat].strip().lower().split(' ')
            tokens = [w.strip("'") for w in tokens if len(w.strip("'")) > 0]

            n = len(tokens)  # type: int
            if n > 5:
                tokens = tokens[:5]
            if n < 5:
                tokens += ['<pad>'] * (5 - n)
        result = tf.py_func(get_content, [line], [tf.string, tf.int32])
        result[0].set_shape([FLAGS.sentence_max_len])
        result[1].set_shape([])
        print('result:{0}')
        # Lookup tokens to return their ids
        ids = vocab.lookup(result[0])



        for feat in numerical_features:
            feature_dict[feat] = parsed[feat]

        for feat in category_features:
            feature_dict[feat] = parsed[feat]

        label = parsed['label']

        return feature_dict, label

    # Extract lines from input files using the Dataset API.
    filenames = tf.data.Dataset.list_files([
        data_path,
    ])
    dataset = filenames.flat_map(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

    batch_size = 2
    dataset = dataset.map(_parse, num_parallel_calls=60)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=batch_size)  # 在缓冲区中随机打乱数据

    # dataset = dataset.padded_batch(batch_size=batch_size,
    #                                padded_shapes=pad_shapes,
    #                                padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    try:
        iterator = dataset.make_one_shot_iterator()
    except AttributeError:
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    return iterator.get_next()


if __name__ == "__main__":
    ########################################################################
    #################数据预处理##############
    ########################################################################
    # 获取char embedding及其查找关系
    embedding_dim = 32
    char_file_names = ['./data/char.json']
    CHAR_ID2IDX, CHAR_EMBEDDING = get_item_embed(char_file_names, embedding_dim)

    # 筛选实体标签categorical 用于定义映射关系
    DICT_CATEGORICAL = {"topic_id": [str(i) for i in range(0, 700)],
                        "keyword_id": [str(i) for i in range(0, 10)],
                        }

    # text_features = ['keyword', 'title', 'brand', 'tag']
    # numerical_features = ['volume']
    # category_features = ['type']
    text_features = []
    # numerical_features = []
    numerical_features = ['volume']
    category_features = []
    _EMBEDDING_DIMENSION = 50

    text_columns = context_feature_columns(text_features).values()

    numerical_columns = [tf.feature_column.numeric_column(feat,default_value=0.0) for feat in
                         numerical_features]

    category_columns = [tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(feat, 1000), 4) for feat in
        category_features]

    # 线性侧特征及交叉侧特征
    dnn_feature_columns = numerical_columns + category_columns
    linear_feature_columns = numerical_columns + category_columns + text_columns

    feature_columns = text_columns + numerical_columns + category_columns
    feature_spec = feature_columns
    # feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    #
    # # 线性侧特征及交叉侧特征
    # linear_feature_columns_name = numerical_features + category_features
    # fm_group_column_name = [col + '_char' for col in text_features] + category_features + numerical_features
    #
    # linear_feature_columns = [col for col in feature_columns if col.name in linear_feature_columns_name]
    # fm_group_columns = [col for col in feature_columns if col.name in fm_group_column_name]

    # DEFAULT_VALUES = [[0], [''], [''], [''], [''], [0.0], [0]]
    DEFAULT_VALUES = [[0.0], [0.0]]
    COL_NAME = ['volume',  'label']

    pad_shapes = {}
    pad_values = {}
    #
    data = pd.read_csv('./hys_df_test.csv', '\t')
    data['label'] = data['label'].apply(float)
    for text in text_features:
        data[text] = pd.DataFrame([(x.split(' ') for x in data[text])], index=data.index)
        # data[text] = data[text].map(lambda x:x.split(' '))
    data.to_csv('./hys_df_test_new.csv', index=False, sep='\t',
                columns=[ 'volume', 'label'])

    #
    batch_size = 2
    # train_input = input_fn('./hys_df_test.csv', shuffle=True, num_epochs=3, batch_size=2)
    train_input = train_input_fn('./hys_df_test_new.csv', feature_spec, 2, 10)
    # 验证集
    val_batch_size = 2
    val_input = train_input_fn('./hys_df_test_new.csv', feature_spec, 1, 10)

    ########################################################################
    #################模型训练##############
    ########################################################################

    # model = DeepFM(feature_columns, dnn_feature_columns=feature_columns, fm_group=feature_columns, task='binary')
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=tf.keras.metrics.AUC(name='auc'))
    #
    #
    #
    # log_dir = './tensorboardshare/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
    #                          histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
    #                          write_graph=True,  # 是否存储网络结构图
    #                          write_images=True,  # 是否可视化参数
    #                          update_freq='epoch',
    #                          embeddings_freq=0,
    #                          embeddings_layer_names=None,
    #                          embeddings_metadata=None,
    #                          profile_batch=20)
    #
    # total_train_sample = 100
    # total_test_sample = 100
    # train_steps_per_epoch = np.floor(total_train_sample / batch_size).astype(np.int32)
    # test_steps_per_epoch = np.ceil(total_test_sample / val_batch_size).astype(np.int32)
    # history_loss = model.fit(train_input, epochs=3,
    #                          steps_per_epoch=train_steps_per_epoch,
    #                          validation_data=val_input, validation_steps=test_steps_per_epoch,
    #                          verbose=1, callbacks=[tbCallBack])
    # model_save_path = os.path.join('./', "deepfm/")
    # tf.saved_model.save(model, model_save_path)

    # 3.Define Model,train,predict and evaluate
    model = DeepFMEstimator(linear_feature_columns, dnn_feature_columns, task='binary')

    model.train(lambda: train_input_fn('./hys_df_test.csv', feature_spec, 2, 10))
    eval_result = model.evaluate(lambda: val_input)

    print(eval_result)
