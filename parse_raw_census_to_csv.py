import csv
import numpy as np
import pandas as pd
import random
import json
import tensorflow as tf
from run_ctr_model import census_text_input_fn_from_tfrecords

ROOT_PATH = './data/'
TRAIN_RAW = ROOT_PATH + 'adult/adult.data'
TEST_RAW = ROOT_PATH + 'adult/adult.test'

MODEL_PATH = '/tmp/adult_model'
EXPORT_PATH = '/tmp/adult_export_model'

EMBEDDING_FEATURE_NAMES = ['user_emb', 'item_emb']
NUMERIC_FEATURE_NAMES = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_FEATURE_WITH_VOCABULARY = {
    'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc',
                  'Without-pay', 'Never-worked'],
    'relationship': ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'],
    'gender': [' Male', 'Female'],
    'marital_status': [' Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated',
                       'Married-AF-spouse', 'Widowed'],
    'race': [' White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    'education': ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc',
                  '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'],
}

CATEGORICAL_FEATURE_WITH_HASH_BUCKETS = {
    'native_country': 60,
    'occupation': 20
}
TEXT_FEATURE_NAMES = ['query', 'title']
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_WITH_VOCABULARY.keys()) + list(
    CATEGORICAL_FEATURE_WITH_HASH_BUCKETS.keys())
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + list(CATEGORICAL_FEATURE_WITH_VOCABULARY.keys()) + list(
    CATEGORICAL_FEATURE_WITH_HASH_BUCKETS.keys())
TARGET_NAME = ['income_bracket']
TARGET_LABELS = [' <=50K', ' >50K']
WEIGHT_COLUMN_NAME = 'fnlwgt'

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket'
]

_STRING_COLS = [
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'native_country',
    'text',
    'income_bracket',
]

_CSV_COLUMN_DEFAULTS = [
    [0], [''], [0], [''], [0],
    [''], [''], [''], [''], [''],
    [0], [0], [0], [''], [0], [0]
]


def label_get(label):
    if label == '>50K':
        return 1
    else:
        return 0


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

    print('item len:{0}'.format(len(item_id)))
    # vocab = tf.contrib.lookup.index_table_from_file(path_vocab, num_oov_buckets=num_oov_buckets)
    # table = tf.lookup.StaticHashTable(
    #     tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
    # out = table.lookup(input_tensor)
    #
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


def create_csv_iterator(csv_file_path, skip_header):
    with tf.gfile.Open(csv_file_path) as csv_file:
        reader = csv.reader(csv_file)
        if skip_header:  # Skip the header
            next(reader)
        for row in reader:
            yield row


embedding_dim = 32
char_file_names = ['./data/char.json']


# CHAR_ID2IDX, CHAR_EMBEDDING = get_item_embed(char_file_names, embedding_dim)
def create_example(row, vocab, header):
    """
    Returns a tensorflow.Example Protocol Buffer object.
    """
    example = tf.train.Example()
    # print('row:{0}'.format(row))
    for i in range(len(header)):

        feature_name = header[i]
        feature_value = row[i]
        if 'emb' in feature_name or 'text' in feature_name:
            print(i, feature_name, feature_value, type(feature_value))

        if feature_name in NUMERIC_FEATURE_NAMES:
            example.features.feature[feature_name].float_list.value.extend([float(feature_value)])

        if feature_name in EMBEDDING_FEATURE_NAMES:
            example.features.feature[feature_name].float_list.value.extend(
                [float(value) for value in feature_value.replace('[', '').replace(']', '').split(',')])

        elif feature_name in CATEGORICAL_FEATURE_NAMES:
            example.features.feature[feature_name].bytes_list.value.extend([bytes(feature_value, 'utf-8')])

        elif feature_name in TEXT_FEATURE_NAMES:

            ids = [bytes(x, 'utf-8') for x in feature_value.split(' ')]
            # ids = [int(x) for x in feature_value.split(' ')]
            example.features.feature[feature_name].bytes_list.value.extend(ids)
            tokens_list = tf.strings.split(['1 2 4 6 0'], ' ')
            ids = vocab.lookup(tokens_list)
            print('ids:{0}'.format(ids))
            # print(tokens_list.values)
            # print(tokens_list.to_tensor(default_value='', shape=[None, 10]))

            # emb = tf.nn.embedding_lookup(params=CHAR_EMBEDDING, ids=CHAR_ID2IDX.lookup(tokens_list))
            # # emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
            # print('first:{0}'.fomart(emb))
            # emb = tf.reduce_mean(emb, axis=0)
            #
            # print('mean:{0}'.fomart(emb))
            # Flatten the sentence embedding back to 1-D.
            # sentence_embedding = tf.reshape(emb, shape=[-1])
            print(feature_value.split(' '))
            # example.features.feature[feature_name].int64_list.value.extend(ids)
            # features = {
            #     # 'id': _bytes_feature(str(record_id)),
            #     'embedding': _float_feature(sentence_embedding.numpy())
            # }

        elif feature_name in TARGET_NAME:
            example.features.feature[feature_name].float_list.value.extend([float(feature_value)])

    return example


def create_tfrecords_file(input_csv_file, header):
    """
    Creates a TFRecords file for the given input data and
    example transofmration function
    """
    output_tfrecord_file = input_csv_file.replace("csv", "tfrecords")
    writer = tf.python_io.TFRecordWriter(output_tfrecord_file)
    vocab = tf.contrib.lookup.index_table_from_file('./char.txt', num_oov_buckets=1)
    print("Creating TFRecords file at", output_tfrecord_file, "...")

    for i, row in enumerate(create_csv_iterator(input_csv_file, skip_header=False)):

        if len(row) == 0:
            continue

        example = create_example(row, vocab, header)
        content = example.SerializeToString()
        writer.write(content)

    writer.close()

    print("Finish Writing", output_tfrecord_file)


def create_embedding_example(text, CHAR_ID2IDX, CHAR_EMBEDDING):
    """Create tf.Example containing the sample's embedding and its ID."""

    tokens_list = tf.strings.split(text)
    # print(tokens_list.values)
    # print(tokens_list.to_tensor(default_value='', shape=[None, 10]))

    emb = tf.nn.embedding_lookup(params=CHAR_EMBEDDING, ids=CHAR_ID2IDX.lookup(tokens_list))
    # emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
    print('first:{0}'.fomart(emb))
    emb = tf.reduce_mean(emb, axis=0)

    print('mean:{0}'.fomart(emb))
    # Flatten the sentence embedding back to 1-D.
    sentence_embedding = tf.reshape(emb, shape=[-1])

    features = {
        # 'id': _bytes_feature(str(record_id)),
        'embedding': _float_feature(sentence_embedding.numpy())
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_embeddings(output_path, starting_record_id):
    record_id = int(starting_record_id)
    embedding_dim = 32
    char_file_names = ['./data/char.json']
    CHAR_ID2IDX, CHAR_EMBEDDING = get_item_embed(char_file_names, embedding_dim)
    text = ['牛', '口']
    with tf.io.TFRecordWriter(output_path) as writer:
        for word in text:
            example = create_embedding_example(word, CHAR_ID2IDX, CHAR_EMBEDDING)
            record_id = record_id + 1
            writer.write(example.SerializeToString())
    return record_id


# create_embeddings('./data/adult/test.tfr', 1)
#
# def census_input_fn_from_tfrecords(data_file, num_epochs, shuffle, batch_size):
#     def _parse_census_TFRecords_fn(record):
#         features = {
#             # int
#             'age': tf.io.FixedLenFeature([], tf.float32),
#             # 'fnlwgt':         tf.io.FixedLenFeature([], tf.float32),
#             'education_num': tf.io.FixedLenFeature([], tf.float32),
#             'capital_gain': tf.io.FixedLenFeature([], tf.float32),
#             'capital_loss': tf.io.FixedLenFeature([], tf.float32),
#             'hours_per_week': tf.io.FixedLenFeature([], tf.float32),
#             # string
#             'gender': tf.io.FixedLenFeature([], tf.string),
#             'education': tf.io.FixedLenFeature([], tf.string),
#             'marital_status': tf.io.FixedLenFeature([], tf.string),
#             'relationship': tf.io.FixedLenFeature([], tf.string),
#             'race': tf.io.FixedLenFeature([], tf.string),
#             'workclass': tf.io.FixedLenFeature([], tf.string),
#             'native_country': tf.io.FixedLenFeature([], tf.string),
#             'occupation': tf.io.FixedLenFeature([], tf.string),
#             'income_bracket': tf.io.FixedLenFeature([], tf.float32),
#             # 'text': tf.io.FixedLenFeature([], tf.string),
#             'text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value='0'),
#             'bert_emb': tf.io.FixedLenFeature([10], tf.float32),  # item向量
#         }
#         features = tf.io.parse_single_example(record, features)
#         # labels = tf.equal(features.pop('income_bracket'), '>50K')
#         # labels = tf.reshape(labels, [-1])
#         # labels = tf.to_float(labels)
#         labels = features.pop('income_bracket')
#         return features, labels
#
#     assert tf.io.gfile.exists(data_file), ('no file named: ' + str(data_file))
#
#     dataset = tf.data.TFRecordDataset(data_file).map(_parse_census_TFRecords_fn, num_parallel_calls=10)
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=5000)
#     dataset = dataset.repeat(num_epochs)
#     dataset = dataset.batch(batch_size)
#     iterator = dataset.make_one_shot_iterator()
#     features, labels = iterator.get_next()
#     return features, labels


def run():
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--type', metavar='N', type=str, choices=["text", "raw"],
                        help='add text')

    args = parser.parse_args()

    ROOT_PATH = './data/'
    # TRAIN_RAW = ROOT_PATH + 'adult/adult.data'
    # TEST_RAW = ROOT_PATH + 'adult/adult.test'
    #
    # HEADER = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
    #           'marital_status', 'occupation', 'relationship', 'race', 'gender',
    #           'capital_gain', 'capital_loss', 'hours_per_week',
    #           'native_country', 'income_bracket']
    #
    # HEADER_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
    #                    [0], [0], [0], [''], ['']]
    # train_df = pd.read_csv(TRAIN_RAW, names=_CSV_COLUMNS)
    # test_df = pd.read_csv(TEST_RAW, names=_CSV_COLUMNS)
    #
    # print(train_df.dtypes)
    # for col in _CSV_COLUMNS:
    #     if col in _STRING_COLS:
    #         print('col:{0}'.format(col))
    #         train_df[col] = train_df[col].map(lambda x: str(x).replace(' ', ''))
    #         test_df[col] = test_df[col].map(lambda x: str(x).replace(' ', ''))
    #
    # if args.type == "text":
    #     lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     print(random.sample(lst, 4))
    #     HEADER = HEADER +TEXT_FEATURE_NAMES+ EMBEDDING_FEATURE_NAMES
    #     # random.randint(3,5)
    #     train_df['query'] = train_df.apply(lambda x: ' '.join([str(x) for x in random.sample(lst, 4)]) + ' 0', axis=1)
    #     test_df['query'] = test_df.apply(lambda x: ' '.join([str(x) for x in random.sample(lst, 4)]) + ' 0', axis=1)
    #     train_df['title'] = train_df.apply(lambda x: ' '.join([str(x) for x in random.sample(lst, 4)]) + ' 0', axis=1)
    #     test_df['title'] = test_df.apply(lambda x: ' '.join([str(x) for x in random.sample(lst, 4)]) + ' 0', axis=1)
    #     train_df['user_emb'] = train_df.apply(lambda x: np.random.uniform(low=-0.1, high=0.1, size=10).tolist(), axis=1)
    #     test_df['user_emb'] = test_df.apply(lambda x: np.random.uniform(low=-0.1, high=0.1, size=10).tolist(), axis=1)
    #     train_df['item_emb'] = train_df.apply(lambda x: np.random.uniform(low=-0.1, high=0.1, size=10).tolist(), axis=1)
    #     test_df['item_emb'] = test_df.apply(lambda x: np.random.uniform(low=-0.1, high=0.1, size=10).tolist(), axis=1)
    #     ROOT_PATH = ROOT_PATH + 'text/'
    # else:
    #     ROOT_PATH = ROOT_PATH + 'raw/'
    # train_df['income_bracket'] = train_df.apply(lambda x: label_get(x.income_bracket), axis=1)
    # test_df['income_bracket'] = test_df.apply(lambda x: label_get(x.income_bracket), axis=1)
    #
    # test_df.sample(frac=1, random_state=2021)
    # eval = test_df.loc[: int(len(test_df) * 0.5)]
    # test = test_df.loc[int(len(test_df) * 0.5) + 1:]
    # test.reset_index(drop=True, inplace=True)
    #
    # TRAIN_PATH = ROOT_PATH + 'adult/train.csv'
    # EVAL_PATH = ROOT_PATH + 'adult/eval.csv'
    # TEST_PATH = ROOT_PATH + 'adult/test.csv'
    # PREDICT_PATH = ROOT_PATH + 'adult/predict.csv'
    #
    # train_df.to_csv(TRAIN_PATH, index=False, header=None)
    # eval.to_csv(EVAL_PATH, index=False, header=None)
    # test.to_csv(TEST_PATH, index=False, header=None)
    #
    # train_data_files = [TRAIN_PATH]
    # valid_data_files = [EVAL_PATH]
    # test_data_files = [TEST_PATH]
    #
    # print("Converting Training Data Files")
    # for input_csv_file in train_data_files:
    #     create_tfrecords_file(input_csv_file,HEADER)
    # print("")
    #
    # print("Converting Validation Data Files")
    # for input_csv_file in valid_data_files:
    #     create_tfrecords_file(input_csv_file,HEADER)
    # print("")
    #
    # print("Converting Test Data Files")
    # for input_csv_file in test_data_files:
    #     create_tfrecords_file(input_csv_file,HEADER)

    print('path:{0}'.format(ROOT_PATH + 'adult/train.tfrecords'))
    train_dataset = census_text_input_fn_from_tfrecords(ROOT_PATH + 'text/adult/train.tfrecords', 1, shuffle=True,
                                                        batch_size=16)
    # iterator = train_dataset.make_one_shot_iterator()
    # element = iterator.get_next()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())

        print('value')
        for i in range(5):
            print('dict:{0}'.format(train_dataset))
            print(session.run(train_dataset))


if __name__ == '__main__':
    run()
