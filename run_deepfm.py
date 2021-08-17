import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from utils.feature_column import get_item_embed
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from deepctr.estimator import DeepFMEstimator


def input_fn(data_path, shuffle, num_epochs, batch_size):
    """Generate an input function for the Estimator."""

    def _parse_function(example_proto):
        item_feats = tf.io.decode_csv(example_proto, record_defaults=DEFAULT_VALUES, field_delim='\t')
        parsed = dict(zip(COL_NAME, item_feats))

        feature_dict = {}
        for feat_col in feature_columns:
            print(parsed)
            if isinstance(feat_col, VarLenSparseFeat):
                # 'sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'
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
                # 'name', 'dimension', 'dtype', 'transform_fn'
                if 'char' in feat_col.name or 'word' in feat_col.name:
                    print('feat_col.name:{0}'.format(parsed[feat_col.name.split('_')[0]]))
                    keys = tf.strings.split(parsed[feat_col.name.split('_')[0]], ' ')
                    print('keys:{0}'.format(keys))
                    emb = tf.nn.embedding_lookup(params=CHAR_EMBEDDING, ids=CHAR_ID2IDX.lookup(keys))
                    # emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
                    emb = tf.reduce_mean(emb, axis=0)
                    feature_dict[feat_col.name] = emb
                elif 'id' in feat_col.name:
                    keys = tf.strings.split(parsed[feat_col.pre_embed], ',')
                    emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(keys))
                    emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
                    feature_dict[feat_col.name] = emb
                else:
                    feature_dict[feat_col.name] = parsed[feat_col.name]

            else:
                raise Exception("unknown feature_columns....")

        label = parsed['label']

        return feature_dict, label

    # Extract lines from input files using the Dataset API.
    filenames = tf.data.Dataset.list_files([
        data_path,
    ])
    dataset = filenames.flat_map(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

    batch_size = 2
    dataset = dataset.map(_parse_function, num_parallel_calls=60)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=batch_size)  # 在缓冲区中随机打乱数据

    # epochs from blending together.
    # dataset = dataset.repeat(num_epochs)
    # dataset = dataset.batch(batch_size) # Batch size to use

    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=pad_shapes,
                                   padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

    # iterator = dataset.make_one_shot_iterator()
    # batch_features, batch_labels = iterator.get_next()
    # # return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    # return batch_features, batch_labels


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

    text_features = ['keyword', 'title', 'brand', 'tag']
    numerical_features = ['volume']
    category_features = ['type']

    text_columns = [DenseFeat(name=feat + '_char', dimension=embedding_dim, ) for
                    feat in text_features]
    numerical_columns = [DenseFeat(name=feat, dimension=1, ) for feat in
                         numerical_features]
    category_columns = [
        SparseFeat(name='type', vocabulary_size=3, embedding_dim=4, dtype='int32') for feat in
        category_features]
    feature_columns = text_columns + numerical_columns + category_columns

    # 线性侧特征及交叉侧特征
    linear_feature_columns_name = numerical_features + category_features
    fm_group_column_name = [col + '_char' for col in text_features] + category_features + numerical_features

    linear_feature_columns = [col for col in feature_columns if col.name in linear_feature_columns_name]
    fm_group_columns = [col for col in feature_columns if col.name in fm_group_column_name]

    DEFAULT_VALUES = [[0], [''], [''], [''], [''], [0.0], [0]]
    COL_NAME = ['label', 'keyword', 'title', 'brand', 'tag', 'volume', 'type']

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
            elif feat_col.dtype == 'int32':
                pad_shapes[feat_col.name] = tf.TensorShape([])
                pad_values[feat_col.name] = 0
            else:
                pad_shapes[feat_col.name] = tf.TensorShape([])
                pad_values[feat_col.name] = 0.0
        elif isinstance(feat_col, DenseFeat):
            # pad_shapes[feat_col.name] = tf.TensorShape([])
            # pad_values[feat_col.name] = 0.0
            # id 或者字符字段都以xxx_id 或者_char _word结尾
            if '_' not in feat_col.name:
                pad_shapes[feat_col.name] = tf.TensorShape([])
                pad_values[feat_col.name] = 0.0
            else:
                pad_shapes[feat_col.name] = tf.TensorShape([feat_col.dimension])
                pad_values[feat_col.name] = 0.0

    pad_shapes = (pad_shapes, (tf.TensorShape([])))
    pad_values = (pad_values, (tf.constant(0, dtype=tf.int32)))

    # filenames = tf.data.Dataset.list_files([
    #     './hys_df_test.csv',
    # ])
    # dataset = filenames.flat_map(
    #     lambda filepath: tf.data.TextLineDataset(filepath).skip(1))
    #
    batch_size = 2
    # dataset = dataset.map(_parse_function, num_parallel_calls=60)
    # dataset = dataset.repeat()
    # dataset = dataset.shuffle(buffer_size=batch_size)  # 在缓冲区中随机打乱数据
    # dataset = dataset.padded_batch(batch_size=batch_size,
    #                                padded_shapes=pad_shapes,
    #                                padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    train_input = input_fn('./hys_df_test.csv', shuffle=True, num_epochs=3, batch_size=2)

    # 验证集
    # filenames_val = tf.data.Dataset.list_files(['./hys_df_test.csv'])
    # dataset_val = filenames_val.flat_map(
    #     lambda filepath: tf.data.TextLineDataset(filepath).skip(1))
    #
    val_batch_size = 2
    # dataset_val = dataset_val.map(_parse_function, num_parallel_calls=60)
    # dataset_val = dataset_val.padded_batch(batch_size=val_batch_size,
    #                                        padded_shapes=pad_shapes,
    #                                        padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
    # dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_input = input_fn('./hys_df_test.csv', shuffle=True, num_epochs=1, batch_size=1)

    ########################################################################
    #################模型训练##############
    ########################################################################
    #
    # model = DeepFM(linear_feature_columns, fm_group_columns, DICT_CATEGORICAL, dnn_hidden_units=(128, 128),
    #                dnn_activation='relu',
    #                seed=1024, )

    model = DeepFM(feature_columns, dnn_feature_columns=feature_columns, fm_group=feature_columns, task='binary')
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=tf.keras.metrics.AUC(name='auc'))

    # 3.Define Model,train,predict and evaluate
    # model = DeepFMEstimator(linear_feature_columns, fm_group_columns, task='binary')

    # model.train(lambda:train_input)
    # eval_result = model.evaluate(lambda:val_input)
    #
    # print(eval_result)

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
    history_loss = model.fit(train_input, epochs=3,
                             steps_per_epoch=train_steps_per_epoch,
                             validation_data=val_input, validation_steps=test_steps_per_epoch,
                             verbose=1, callbacks=[tbCallBack])
    model_save_path = os.path.join('./', "deepfm/")
    tf.saved_model.save(model, model_save_path)
