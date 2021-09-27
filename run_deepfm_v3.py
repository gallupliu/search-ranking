import os
import json
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_text as tf_text
from deepctr.estimator import DeepFMEstimator
from deepctr.estimator.inputs import input_fn_tfrecord


def test_data():
    hsy_data = {
        "keyword": ["安 慕 希", "牛 奶", "牛", "奶 粉", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
        "title": ["安 慕 希", "牛 奶", "牛", "奶 粉", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
        "brand": ["安 慕 希", "伊 利", "蒙 牛", "奶 粉", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
        "tag": ["酸 奶", "纯 牛 奶", "牛", "固 态 奶", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
        "volume": [1, 2, 3, 4, 5, 4.3, 1.2, 4.5, 1.0, 0.8],
        "type": [0, 1, 0, 1, 2, 1, 0, 0, 2, 1],
        "label": [0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    }
    hsy_df = pd.DataFrame(hsy_data)
    # print(hsy_df.head(10))
    hsy_df.to_csv('./milk.csv', index=False, sep='\t')


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


# def train_input_fn(data_path, feature_spec, shuffle, num_epochs, batch_size):
#     # Extract lines from input files using the Dataset API.
#     filenames = tf.data.Dataset.list_files([
#         data_path,
#     ])
#     dataset = filenames.flat_map(
#         lambda filepath: tf.data.TextLineDataset(filepath).skip(1))
#
#     tokenizer = tf_text.WhitespaceTokenizer()
#
#     def _parse(example):
#         columns = tf.io.decode_csv(example, record_defaults=record_defaults, field_delim='\t')
#         parsed = dict(zip(COL_NAME, columns))
#
#         feature_dict = {}
#         # for column in TEXT_COLUMNS:
#             # tokens_list = tf.strings.split([parsed[column]])
#             # print(tokens_list.values)
#             # print(tokens_list.to_tensor(default_value='', shape=[None, 10]))
#             # tokens_list = tf.strings.split(parsed[column])
#             # tokens_list = parsed[column]
#             # emb = tf.nn.embedding_lookup(params=CHAR_EMBEDDING, ids=CHAR_ID2IDX.lookup(tokens_list))
#             # print[emb]
#             # emb = tf.reshape(emb, shape=[-1])
#             # emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
#             # emb = tf.reduce_mean(emb, 0)
#             # emb = tf.reshape(emb, shape=[-1])
#             # emb = tf.reduce_mean(emb, axis=0)
#             # print('first emb:{0}'.format(emb))
#             # emb = tf.reduce_mean(emb, axis=0)
#             # emb = tf.reshape(emb, shape=[-1])
#             # print('first emb:{0}'.format(emb))
#             # feature_dict[column + '_emb'] = tokens_list
#             # # tokens = tokenizer.tokenize(parsed['keyword'])
#             # tokens = tokens_list.to_tensor(default_value='', shape=[None, 10])
#             # cur_list_size = tokens.shape[0]
#             # print('first')
#             # print(tokens, cur_list_size)
#             # # ids = vocab.lookup(tokens)
#             # ids = CHAR_ID2IDX.lookup(tokens)
#             # feature_dict[column + '_tokens'] = tokens
#             # feature_dict[column + '_ids'] = ids
#             # # ids = tf.reshape(ids, [1, -1])
#             # # print(ids.to_tensor(0))
#             #
#             # print('ids:{0} {1} {2}'.format(ids, ids.shape[0], tf.reshape(ids, [1, -1])))
#
#         # for column in NUMERICAL_COLUMNS:
#         #     print('feat_name:{0},value:{1}'.format(column, parsed[column]))
#         #     feature_dict[column] = parsed[column]
#         #
#         # for column in CATEGORY_COLUMNS:
#         #     print('feat_name:{0},value:{1}'.format(column, parsed[column]))
#         #     feature_dict[column] = parsed[column]
#         labels = parsed['label']
#         feature_dict['label'] = parsed['label']
#         spec = tf.feature_column.make_parse_example_spec(feature_spec)
#         #
#         spec.update({"label": tf.io.FixedLenFeature([], tf.int64)})
#         spec["label"] = tf.io.FixedLenFeature([], tf.int32)
#         print('spec:{0}'.format(spec))
#         print('example:{0}'.format(example))
#         feats = tf.io.parse_example(example, features=spec)
#         labels = feats.pop('label')
#         return feature_dict, labels
#
#     dataset = dataset.map(lambda x: _parse(x))
#     dataset = dataset.shuffle(buffer_size=2500,reshuffle_each_iteration=True)
#     dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
#     return dataset

def train_input_fn(data_path, feature_spec, shuffle, num_epochs, batch_size):
    # Extract lines from input files using the Dataset API.
    filenames = tf.data.Dataset.list_files([
        data_path,
    ])
    dataset = filenames.flat_map(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

    # tokenizer = tf_text.WhitespaceTokenizer()

    def _parse(example):
        columns = tf.io.decode_csv(example, record_defaults=record_defaults, field_delim='\t')
        parsed = dict(zip(COL_NAME, columns))
        print('columns:{0}'.format(columns))

        feature_dict = {}
        for column in TEXT_COLUMNS:
            tokens_list = tf.strings.split([parsed[column]])
            print(tokens_list.values)
            print(tokens_list.to_tensor(default_value='', shape=[None, 10]))
            tokens_list = tf.strings.split([parsed[column]])
            # tokens_list = parsed[column]
            # emb = tf.nn.embedding_lookup(params=CHAR_EMBEDDING, ids=CHAR_ID2IDX.lookup(tokens_list))
            # print[emb]
            # emb = tf.reshape(emb, shape=[-1])
            # # emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
            # emb = tf.reduce_mean(emb, 0)
            # emb = tf.reshape(emb, shape=[-1])
            # emb = tf.reduce_mean(emb, axis=0)
            # print('first emb:{0}'.format(emb))
            # emb = tf.reduce_mean(emb, axis=0)
            # emb = tf.reshape(emb, shape=[-1])
            # print('first emb:{0}'.format(emb))
            # feature_dict[column + '_emb'] = tokens_list
            # tokens = tokenizer.tokenize(parsed['keyword'])
            tokens = tokens_list.to_tensor(default_value='', shape=[None, 10])
            cur_list_size = tokens.shape[0]
            print('first')
            print(tokens, cur_list_size)
            # ids = vocab.lookup(tokens)
            ids = CHAR_ID2IDX.lookup(tokens)
            # feature_dict[column + '_tokens'] = tokens
            feature_dict[column + '_ids'] = ids
            # ids = tf.reshape(ids, [1, -1])
            # print(ids.to_tensor(0))

            print('ids:{0} {1} {2}'.format(ids, ids.shape[0], tf.reshape(ids, [1, -1])))

        for column in NUMERICAL_COLUMNS:
            print('feat_name:{0},value:{1}'.format(column, parsed[column]))
            feature_dict[column] = parsed[column]

        for column in CATEGORY_COLUMNS:
            print('feat_name:{0},value:{1}'.format(column, parsed[column]))
            feature_dict[column] = parsed[column]
        labels = parsed['label']
        # feature_dict['label'] = parsed['label']
        # spec = tf.feature_column.make_parse_example_spec(feature_spec)
        # #
        # spec.update({"label": tf.io.FixedLenFeature([], tf.int64)})
        # spec["label"] = tf.io.FixedLenFeature([], tf.int32)
        # print('spec:{0}'.format(spec))
        # print('example:{0}'.format(example))
        # feats = tf.io.parse_example(example, features=spec)
        # labels = feats.pop('label')
        return feature_dict, labels

    dataset = dataset.map(lambda x: _parse(x))
    dataset = dataset.shuffle(buffer_size=2500,reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    return dataset


def _int64_feature(value):
  """Returns int64 tf.train.Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))


def _bytes_feature(value):
  """Returns bytes tf.train.Feature."""
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def _float_feature(value):
  """Returns float tf.train.Feature."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))


def create_embedding_example(text,CHAR_ID2IDX, CHAR_EMBEDDING):
  """Create tf.Example containing the sample's embedding and its ID."""

  tokens_list = tf.strings.split(text)
  # print(tokens_list.values)
  # print(tokens_list.to_tensor(default_value='', shape=[None, 10]))

  emb = tf.nn.embedding_lookup(params=CHAR_EMBEDDING, ids=CHAR_ID2IDX.lookup(tokens_list))
  # emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
  emb = tf.reduce_mean(emb, axis=0)

  print(emb)
  # Flatten the sentence embedding back to 1-D.
  sentence_embedding = tf.reshape(emb, shape=[-1])

  features = {
      # 'id': _bytes_feature(str(record_id)),
      'embedding': _float_feature(sentence_embedding.numpy())
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


def create_embeddings( output_path, starting_record_id):
  record_id = int(starting_record_id)
  text = ['牛 奶', '口 味']
  with tf.io.TFRecordWriter(output_path) as writer:
    for word in text:
      example = create_embedding_example(word,CHAR_ID2IDX, CHAR_EMBEDDING)
      record_id = record_id + 1
      writer.write(example.SerializeToString())
  return record_id



def create_new_csv(file_path,columns,text_features):
    data = pd.read_csv(file_path, '\t')

    for text in text_features:
        data[text] = pd.DataFrame([(x.split(' ') for x in data[text])], index=data.index)
        # data[text] = data[text].map(lambda x:x.split(' '))
    for column in columns:
        if column not in text_features:
            data[column] = data[column].apply(float)
    data['label'] = data['label'].apply(float)
    data.to_csv('./milk_new.csv', index=False, sep='\t',
                columns=columns)

if __name__ == "__main__":
    # test_data()
    file_path = './milk.csv'


    ########################################################################
    #################数据预处理##############
    ########################################################################
    # 获取char embedding及其查找关系
    TEXT_COLUMNS = ["keyword", "title", "brand", "tag"]
    NUMERICAL_COLUMNS = ["volume"]
    CATEGORY_COLUMNS = ["type"]

    COL_NAME = ["keyword", "title", "brand", "tag", "volume", "type", "label"]
    record_defaults = [[""], [""], [""], [""], [0.0], [0], [0]]
    embedding_dim = 32
    char_file_names = ['./data/char.json']
    CHAR_ID2IDX, CHAR_EMBEDDING = get_item_embed(char_file_names, embedding_dim)
    create_new_csv(file_path, COL_NAME, TEXT_COLUMNS)
    file_path = './milk.csv'
    # Persist TF.Example features containing embeddings for training data in
    # TFRecord format.
    # create_embeddings('./embeddings.tfr', 0)


    # text_features = ['keyword', 'title', 'brand', 'tag']
    # numerical_features = ['volume']
    # category_features = ['type']
    # text_features = []
    # # numerical_features = []
    # numerical_features = ['volume']
    # category_features = []
    _EMBEDDING_DIMENSION = 50

    # text_columns = context_feature_columns(TEXT_COLUMNS).values()
    text_columns = [tf.feature_column.numeric_column(feat + '_emb', default_value=0.0) for feat in
                    TEXT_COLUMNS]
    numerical_columns = [tf.feature_column.numeric_column(feat, default_value=0.0) for feat in
                         NUMERICAL_COLUMNS]

    category_columns = [tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(feat, 1000), 4) for feat in
        CATEGORY_COLUMNS]

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
    # DEFAULT_VALUES = [[0.0], [0.0]]
    # COL_NAME = ['volume', 'label']
    #
    # pad_shapes = {}
    # pad_values = {}
    # #
    # data = pd.read_csv('./hys_df_test.csv', '\t')
    # data['label'] = data['label'].apply(float)
    # for text in TEXT_COLUMNS:
    #     data[text] = pd.DataFrame([(x.split(' ') for x in data[text])], index=data.index)
    #     # data[text] = data[text].map(lambda x:x.split(' '))
    # data.to_csv('./hys_df_test_new.csv', index=False, sep='\t',
    #             columns=['volume', 'label'])

    #
    # batch_size = 2
    # # train_input = input_fn('./hys_df_test.csv', shuffle=True, num_epochs=3, batch_size=2)
    # train_input = train_input_fn(file_path, feature_spec,shuffle=True, num_epochs=1, batch_size=3)
    # iterator = iter(train_input)
    # print('value')
    # print(next(iterator))
    # print(next(iterator))
    # # print(next(iterator))
    # # # 验证集
    # val_batch_size = 2
    # val_input = train_input_fn(file_path, feature_spec,shuffle=False,  num_epochs=1, batch_size=2)

    ########################################################################
    #################模型训练##############
    ########################################################################

    # model = DeepFM(feature_columns, dnn_feature_columns=feature_columns, fm_group=feature_columns, task='binary')
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=tf.keras.metrics.AUC(name='auc'))
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
    # from tensorflow.python.framework.ops import disable_eager_execution
    #
    # disable_eager_execution()
    # # 3.Define Model,train,predict and evaluate
    # model = DeepFMEstimator(linear_feature_columns, dnn_feature_columns, task='binary')
    #
    # model.train(lambda: train_input_fn(file_path, feature_spec,shuffle=True, num_epochs=1, batch_size=3))
    # eval_result = model.evaluate(lambda: train_input_fn(file_path, feature_spec,shuffle=False,  num_epochs=1, batch_size=2))
    #
    # print(eval_result)
