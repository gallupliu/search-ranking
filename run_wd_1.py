# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def build_model_columns(type):
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
    #
    # text = tf.feature_column.categorical_column_with_vocabulary_list(
    #     'text', [
    #         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # To show an example of hashing:
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)

    # Transformations.
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    if type == 'text':
        text_column =  tf.feature_column.categorical_column_with_vocabulary_file(
            key="text",
            vocabulary_file='./ids.txt',
            num_oov_buckets=0)

    # Wide columns and deep columns.
    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=_HASH_BUCKET_SIZE),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'],
            hash_bucket_size=_HASH_BUCKET_SIZE),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        # tf.feature_column.indicator_column(text),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8),

        # columns = shared_embedding_columns(
        # [watched_video_id, impression_video_id], dimension=10)
    ]
    if type == 'text':
        deep_columns.append(tf.feature_column.embedding_column(text_column, 10, combiner='sum'))

    return wide_columns, deep_columns


def input_fn(data_path, shuffle, num_epochs, batch_size):
    """Generate an input function for the Estimator."""
    _CSV_COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income_bracket'
    ]

    _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                            [0], [0], [0], [''], [0]]

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        # text = features.pop('text')
        # text_feat = tf.string_split([text],' ')

        # feat = tf.string_split(columns.values[1:],":")
        # feat = tf.reshape(text_feat .values,text_feat.dense_shape)
        # feat_id,feat_val = tf.split(feat,num_or_size_splits=2,axis=1)
        # feat_id = tf.string_to_number(feat_id,out_type=tf.int32)
        # feat_val = tf.string_to_number(feat_val,out_type=tf.float32)
        # features['text'] = feat_id
        # features['text'] = tf.sparse_tensor_to_dense(features['text_tokens'], default_value='0')
        # features['text'] = tf.string_to_number(features['text'])
        # print(features['text'])
        # print(tf.string_to_number(features['text']))
        # features.pop('text')
        # features.pop('text_tokens')
        labels = features.pop('income_bracket')
        # labels = tf.equal(labels, '>50K')  # binary classification
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_path)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # dataset = dataset.padded_batch(4, padded_shapes=([vectorSize], [None]))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)


    return dataset


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
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=5000)
#     dataset = dataset.repeat(num_epochs)
#     dataset = dataset.batch(batch_size)
#     # iterator = dataset.make_one_shot_iterator()
#     # features, labels = iterator.get_next()
#     # iterator = dataset.make_one_shot_iterator()
#     # features, labels = iterator.get_next()
#     return dataset

def census_input_fn_from_tfrecords(data_file, num_epochs, shuffle, batch_size):
    def _parse_census_TFRecords_fn(record):
        features = {
            # int
            'age': tf.io.FixedLenFeature([], tf.float32),
            # 'fnlwgt':         tf.io.FixedLenFeature([], tf.float32),
            'education_num': tf.io.FixedLenFeature([], tf.float32),
            'capital_gain': tf.io.FixedLenFeature([], tf.float32),
            'capital_loss': tf.io.FixedLenFeature([], tf.float32),
            'hours_per_week': tf.io.FixedLenFeature([], tf.float32),
            # string
            'gender': tf.io.FixedLenFeature([], tf.string),
            'education': tf.io.FixedLenFeature([], tf.string),
            'marital_status': tf.io.FixedLenFeature([], tf.string),
            'relationship': tf.io.FixedLenFeature([], tf.string),
            'race': tf.io.FixedLenFeature([], tf.string),
            'workclass': tf.io.FixedLenFeature([], tf.string),
            'native_country': tf.io.FixedLenFeature([], tf.string),
            'occupation': tf.io.FixedLenFeature([], tf.string),
            'income_bracket': tf.io.FixedLenFeature([], tf.float32),
            # 'text': tf.io.FixedLenFeature([], tf.string),
            'text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value='0'),
        }
        features = tf.io.parse_single_example(record, features)
        # labels = tf.equal(features.pop('income_bracket'), '>50K')
        # labels = tf.reshape(labels, [-1])
        # labels = tf.to_float(labels)
        labels = features.pop('income_bracket')
        return features, labels

    assert tf.io.gfile.exists(data_file), ('no file named: ' + str(data_file))

    dataset = tf.data.TFRecordDataset(data_file).map(_parse_census_TFRecords_fn, num_parallel_calls=10)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels



# estimator.train()可以循环运行，模型的状态将持久保存在model_dir
def run():
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--type', metavar='N', type=str, choices=["text", "raw"],
                        help='add text')

    args = parser.parse_args()


    if args.type == "text":
        MODEL_PATH = './tmp/text/adult_model'
        EXPORT_PATH = './tmp/text/adult_export_model'
        ROOT_PATH = './data/text/adult/'
        TRAIN_PATH = ROOT_PATH + 'train.csv'
        EVAL_PATH = ROOT_PATH + 'train.csv'
        PREDICT_PATH = ROOT_PATH + 'train.csv'

        _CSV_COLUMNS = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'gender',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
            'income_bracket', 'text'
        ]
        _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                                [0], [0], [0], [''], [0], ['']]
        # dataset = census_input_fn_from_tfrecords(data_file=TRAIN_PATH, num_epochs=40, shuffle=True, batch_size=32)
        # dataset = census_input_fn_from_tfrecords(ROOT_PATH + 'train.tfrecords', 1, shuffle=True, batch_size=32)
        # iterator = dataset.make_one_shot_iterator()
        # element = iterator.get_next()
        # with tf.Session() as session:
        #     session.run(tf.global_variables_initializer())
        #     session.run(tf.tables_initializer())
        #
        #     print('value')
        #     for i in range(5):
        #         print(element)
        #         print(session.run(element))

        dataset = census_input_fn_from_tfrecords(ROOT_PATH + 'train.tfrecords', 1, shuffle=True,
                                                       batch_size=32)
        # iterator = dataset.make_one_shot_iterator()
        # element = iterator.get_next()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            print('value')
            for i in range(5):
                print(dataset)
                print(session.run(dataset))
    else:
        MODEL_PATH = './tmp/raw/adult_model'
        EXPORT_PATH = './tmp/raw/adult_export_model'
        ROOT_PATH = './data/raw/adult/'
        TRAIN_PATH = ROOT_PATH + 'train.csv'
        EVAL_PATH = ROOT_PATH + 'train.csv'
        PREDICT_PATH = ROOT_PATH + 'train.csv'
        _CSV_COLUMNS = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'gender',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
            'income_bracket'
        ]

        _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                                [0], [0], [0], [''], [0]]

        dataset = input_fn(data_path=TRAIN_PATH, shuffle=True, num_epochs=40, batch_size=100)
        iterator = dataset.make_one_shot_iterator()
        element = iterator.get_next()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            print('value')
            for i in range(5):
                print(element)
                print(session.run(element))

    wide_columns, deep_columns = build_model_columns(args.type)


    os.system('rm -rf {}'.format(MODEL_PATH))
    config = tf.estimator.RunConfig(save_checkpoints_steps=100)
    estimator = tf.estimator.DNNLinearCombinedClassifier(model_dir=MODEL_PATH,
                                                         linear_feature_columns=wide_columns,
                                                         linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.01),
                                                         dnn_feature_columns=deep_columns,
                                                         dnn_hidden_units=[256, 64, 32, 16],
                                                         dnn_optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                         config=config)
    # Linear model.
    # estimator = tf.estimator.LinearClassifier(feature_columns=wide_columns, n_classes=2,
    #                                           optimizer=tf.train.FtrlOptimizer(learning_rate=0.03))

    if args.type == "raw":
        # Train the model.
        estimator.train(
            input_fn=lambda: input_fn(data_path=TRAIN_PATH, shuffle=True, num_epochs=40, batch_size=100), steps=2000)
        """
        steps: 最大训练次数，模型训练次数由训练样本数量、num_epochs、batch_size共同决定，通过steps可以提前停止训练
        """
        # Evaluate the model.
        eval_result = estimator.evaluate(
            input_fn=lambda: input_fn(data_path=EVAL_PATH, shuffle=False, num_epochs=1, batch_size=40))

        print('Test set accuracy:', eval_result)

        # Predict.
        pred_dict = estimator.predict(
            input_fn=lambda: input_fn(data_path=PREDICT_PATH, shuffle=False, num_epochs=1, batch_size=40))
    else:
        # Train the model.
        estimator.train(
            input_fn=lambda: census_input_fn_from_tfrecords(ROOT_PATH + 'train.tfrecords', 1, shuffle=True,
                                                       batch_size=32), steps=2000)
        """
        steps: 最大训练次数，模型训练次数由训练样本数量、num_epochs、batch_size共同决定，通过steps可以提前停止训练
        """
        # Evaluate the model.
        eval_result = estimator.evaluate(
            input_fn=lambda: census_input_fn_from_tfrecords(ROOT_PATH + 'eval.tfrecords', 1, shuffle=False,
                                                       batch_size=32))

        print('Test set accuracy:', eval_result)

        # Predict.
        pred_dict = estimator.predict(
            input_fn=lambda: census_input_fn_from_tfrecords(ROOT_PATH + 'test.tfrecords', 1, shuffle=False,
                                                       batch_size=32))

    for pred_res in pred_dict:
        print(pred_res['probabilities'][1])

    columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=columns)
    print('feature_spec:{0}'.format(feature_spec))
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_savedmodel(EXPORT_PATH, serving_input_fn)


if __name__ == '__main__':
    run()
