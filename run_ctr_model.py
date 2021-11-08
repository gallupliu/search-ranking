import argparse
import os
import shutil
import yaml
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from config.feature.rank.census_ctr_feat_config_v1 import CENSUS_CONFIG, build_census_feat_columns
from config.feature.rank.hys_ctr_feat_config_v1 import build_hys_feat_columns
from models.wdl_v1 import wdl_estimator
from models.deepfm_v1 import deepfm_model_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def census_input_fn_from_csv_file(data_file, num_epochs, shuffle, batch_size):
    def _parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=CENSUS_CONFIG['columns_defaults'])
        features = dict(zip(CENSUS_CONFIG['columns'], columns))
        print(CENSUS_CONFIG['columns'])
        print('columns:{0}'.format(columns))
        labels = tf.equal(features.pop('income_bracket'), '>50K')
        labels = tf.reshape(labels, [-1])
        labels = tf.to_float(labels)
        return features, labels

    assert tf.io.gfile.exists(data_file), ('no file named : ' + str(data_file))
    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(_parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def census_input_fn_from_tfrecords(data_file, num_epochs, shuffle, batch_size):
    def _parse_census_TFRecords_fn(record):
        features = {
            # int
            'age': tf.io.FixedLenFeature([], tf.int64),
            'fnlwgt': tf.io.FixedLenFeature([], tf.int64),
            'education_num': tf.io.FixedLenFeature([], tf.int64),
            'capital_gain': tf.io.FixedLenFeature([], tf.int64),
            'capital_loss': tf.io.FixedLenFeature([], tf.int64),
            'hours_per_week': tf.io.FixedLenFeature([], tf.int64),
            # string
            'gender': tf.io.FixedLenFeature([], tf.string),
            'education': tf.io.FixedLenFeature([], tf.string),
            'marital_status': tf.io.FixedLenFeature([], tf.string),
            'relationship': tf.io.FixedLenFeature([], tf.string),
            'race': tf.io.FixedLenFeature([], tf.string),
            'workclass': tf.io.FixedLenFeature([], tf.string),
            'native_country': tf.io.FixedLenFeature([], tf.string),
            'occupation': tf.io.FixedLenFeature([], tf.string),
            'income_bracket': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(record, features)
        labels = tf.equal(features.pop('income_bracket'), '>50K')
        labels = tf.reshape(labels, [-1])
        labels = tf.to_float(labels)
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


def hys_input_fn_from_tfrecords(data_file, num_epochs, shuffle, batch_size):
    def _parse_census_TFRecords_fn(record):
        features = {
            # int
            "id": tf.io.FixedLenFeature([], tf.int64),
            # string
            # "keyword": tf.io.FixedLenFeature([5], tf.string),
            # "title": tf.io.FixedLenFeature([5], tf.string),
            # "brand": tf.io.FixedLenFeature([5], tf.string),
            # "tag": tf.io.FixedLenFeature([5], tf.string),
            "text": tf.io.FixedLenFeature([20], tf.string),
            "type": tf.io.FixedLenFeature([], tf.string),

            "volume": tf.io.FixedLenFeature([], tf.float32),
            "price": tf.io.FixedLenFeature([], tf.float32),
            'user_bert_emb': tf.io.FixedLenFeature([10], tf.float32),  # query向量
            'item_bert_emb': tf.io.FixedLenFeature([10], tf.float32),  # item向量
            "label": tf.io.FixedLenFeature([], tf.int64),

        }
        features = tf.io.parse_single_example(record, features)
        labels = features.pop('label')
        return features, labels

    # tf.compat.v1.gfile.Glob(path2)
    print(tf.io.gfile.listdir)
    # assert tf.io.gfile.exists(tf.io.gfile.glob(data_file)), ('no file named: ' + str(data_file))
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(data_file)).map(_parse_census_TFRecords_fn,
                                                                       num_parallel_calls=10)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    labels = tf.compat.v1.to_float(labels)
    return features, labels


def census_text_input_fn_from_tfrecords(data_file, num_epochs, shuffle, batch_size):
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
            'query': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value='0'),
            'title': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value='0'),
            'user_emb': tf.io.FixedLenFeature([10], tf.float32),  # item向量
            'item_emb': tf.io.FixedLenFeature([10], tf.float32),  # item向量
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


def build_estimator(ckpt_dir, model_name, params_config):
    model_fn_map = params_config['model_fn_map']
    assert model_name in model_fn_map.keys(), ('no model named : ' + str(model_name))
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}),
        save_checkpoints_steps=2000,
        save_summary_steps=500,
        log_step_count_steps=500,
        keep_checkpoint_max=3
    )
    if model_name == 'wdl':
        return wdl_estimator(params=params_config, config=run_config)
    else:
        return tf.estimator.Estimator(model_fn=model_fn_map[model_name], model_dir=ckpt_dir, config=run_config,
                                      params=params_config)


def train_census_data(ARGS):
    feat_columns = build_census_feat_columns(emb_dim=8)
    CENSUS_PATH = './pb/census/'
    MODEL_FN_MAP = {
        'wdl': wdl_estimator,
        # 'dcn':      dcn_model_fn,
        # 'autoint':  autoint_model_fn,
        # 'xdeepfm':  xdeepfm_model_fn,
        'deepfm': deepfm_model_fn,
        # 'resnet':   res_model_fn,
        # 'pnn':      pnn_model_fn,
        # 'fibinet':  fibinet_model_fn,
        # 'afm':      afm_model_fn,
    }

    print(ARGS['model_name'])
    ARGS['ckpt_dir'] = CENSUS_PATH + 'ckpt_dir/' + ARGS['model_name']
    ARGS['embedding_dim'] = feat_columns['embedding_dim']
    ARGS['deep_columns'] = feat_columns['deep_columns']
    ARGS['deep_fields_size'] = feat_columns['deep_fields_size']
    ARGS['wide_columns'] = feat_columns['deep_columns']
    ARGS['wide_fields_size'] = feat_columns['wide_fields_size']
    ARGS['emb_columns'] = feat_columns['emb_columns']
    ARGS['emb_fields_size'] = feat_columns['text_fields_size']
    ARGS['text_columns'] = feat_columns['text_columns']
    ARGS['text_fields_size'] = feat_columns['emb_fields_size']
    ARGS['model_fn_map'] = MODEL_FN_MAP
    print("emb_columns:{0},text_columns:{1}".format(feat_columns['emb_columns'], feat_columns['text_columns']))
    print('this process will train a: ' + ARGS['model_name'] + ' model...')
    print('args:{0}'.format(ARGS))
    shutil.rmtree(ARGS['ckpt_dir'] + '/' + ARGS['model_name'] + '/', ignore_errors=True)
    model = build_estimator(ARGS['ckpt_dir'], ARGS['model_name'], params_config=ARGS)

    # dataset = census_input_fn_from_csv_file(
    #     data_file=ARGS['train_data_dir'],
    #     num_epochs=ARGS['train_epoches_num'],
    #     shuffle=True if ARGS['shuffle'] == True else False,
    #     batch_size=ARGS['batch_size']
    # )
    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())
    #     session.run(tf.tables_initializer())
    #
    #     print('value')
    #     for i in range(5):
    #         print(dataset)
    #         print(session.run(dataset))
    EXPORT_PATH = CENSUS_PATH + ARGS.get('model_name') + '/'
    if not ARGS.get('load_tf_records_data'):

        model.train(
            input_fn=lambda: census_input_fn_from_csv_file(
                data_file=ARGS['train_data_dir'],
                num_epochs=ARGS['train_epoches_num'],
                shuffle=True if ARGS['shuffle'] == True else False,
                batch_size=ARGS['batch_size']
            )
        )
        results = model.evaluate(
            input_fn=lambda: census_input_fn_from_csv_file(
                data_file=ARGS['valid_data_dir'],
                num_epochs=1,
                shuffle=False,
                batch_size=ARGS['batch_size']
            )
        )
        for key in sorted(results):
            print('%s: %s' % (key, results[key]))

        pred_dict = model.predict(
            input_fn=lambda: census_input_fn_from_csv_file(
                data_file=ARGS['test_data_dir'],
                num_epochs=1,
                shuffle=False,
                batch_size=ARGS['batch_size']
            )
        )
        for pred_res in pred_dict:
            print(pred_res)
            if ARGS['model_name'] == 'wdl':
                print(pred_res['classes'][0], pred_res['probabilities'][0])
            else:
                print(pred_res['label'][0], pred_res['probabilities'][0])

    else:
        print('train_data_tfrecords_dir:{0}'.format(ARGS['train_data_tfrecords_dir']))
        model.train(
            input_fn=lambda: census_text_input_fn_from_tfrecords(
                data_file=ARGS['train_data_tfrecords_dir'],
                num_epochs=ARGS['train_epoches_num'],
                shuffle=True,
                batch_size=ARGS['batch_size']
            )
        )
        results = model.evaluate(
            input_fn=lambda: census_text_input_fn_from_tfrecords(
                data_file=ARGS['test_data_tfrecords_dir'],
                num_epochs=1,
                shuffle=False,
                batch_size=ARGS['batch_size']
            )
        )
        for key in sorted(results):
            print('%s: %s' % (key, results[key]))

        predictions = model.predict(
            input_fn=lambda: census_input_fn_from_tfrecords(data_file=ARGS['test_data_tfrecords_dir'], num_epochs=1,
                                                            shuffle=False, batch_size=ARGS['batch_size'])
        )
        # for x in predictions:
        #     print(x['probabilities'][0])
        #     print(x['label'][0]))

        columns = ARGS['wide_columns'] + ARGS['deep_columns'] + ARGS['text_columns'] + ARGS['emb_columns']
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=columns)
        print('feature_spec:{0}'.format(feature_spec))
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel(EXPORT_PATH, serving_input_fn)


def train_hys_data(ARGS):
    feat_columns = build_hys_feat_columns(emb_dim=8)
    CENSUS_PATH = './pb/hys/'
    MODEL_FN_MAP = {
        'wdl': wdl_estimator,
        # 'dcn':      dcn_model_fn,
        # 'autoint':  autoint_model_fn,
        # 'xdeepfm':  xdeepfm_model_fn,
        'deepfm': deepfm_model_fn,
        # 'resnet':   res_model_fn,
        # 'pnn':      pnn_model_fn,
        # 'fibinet':  fibinet_model_fn,
        # 'afm':      afm_model_fn,
    }

    print(ARGS['model_name'])
    ARGS['ckpt_dir'] = CENSUS_PATH + 'ckpt_dir/' + ARGS['model_name']
    ARGS['embedding_dim'] = feat_columns['embedding_dim']
    ARGS['deep_columns'] = feat_columns['deep_columns']
    ARGS['deep_fields_size'] = feat_columns['deep_fields_size']
    ARGS['wide_columns'] = feat_columns['deep_columns']
    ARGS['wide_fields_size'] = feat_columns['wide_fields_size']
    ARGS['emb_columns'] = feat_columns['emb_columns']
    ARGS['emb_fields_size'] = feat_columns['text_fields_size']
    ARGS['text_columns'] = feat_columns['text_columns']
    ARGS['text_fields_size'] = feat_columns['emb_fields_size']
    ARGS['model_fn_map'] = MODEL_FN_MAP
    print('this process will train a: ' + ARGS['model_name'] + ' model...')
    print('args:{0}'.format(ARGS))
    shutil.rmtree(ARGS['ckpt_dir'] + '/' + ARGS['model_name'] + '/', ignore_errors=True)
    model = build_estimator(ARGS['ckpt_dir'], ARGS['model_name'], params_config=ARGS)

    # dataset = census_input_fn_from_csv_file(
    #     data_file=ARGS['train_data_dir'],
    #     num_epochs=ARGS['train_epoches_num'],
    #     shuffle=True if ARGS['shuffle'] == True else False,
    #     batch_size=ARGS['batch_size']
    # )
    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())
    #     session.run(tf.tables_initializer())
    #
    #     print('value')
    #     for i in range(5):
    #         print(dataset)
    #         print(session.run(dataset))
    EXPORT_PATH = CENSUS_PATH + ARGS.get('model_name') + '/'
    if not ARGS.get('load_tf_records_data'):

        model.train(
            input_fn=lambda: census_input_fn_from_csv_file(
                data_file=ARGS['train_data_dir'],
                num_epochs=ARGS['train_epoches_num'],
                shuffle=True if ARGS['shuffle'] == True else False,
                batch_size=ARGS['batch_size']
            )
        )
        results = model.evaluate(
            input_fn=lambda: census_input_fn_from_csv_file(
                data_file=ARGS['valid_data_dir'],
                num_epochs=1,
                shuffle=False,
                batch_size=ARGS['batch_size']
            )
        )
        for key in sorted(results):
            print('%s: %s' % (key, results[key]))

        pred_dict = model.predict(
            input_fn=lambda: census_input_fn_from_csv_file(
                data_file=ARGS['test_data_dir'],
                num_epochs=1,
                shuffle=False,
                batch_size=ARGS['batch_size']
            )
        )
        for pred_res in pred_dict:
            print(pred_res)
            if ARGS['model_name'] == 'wdl':
                print(pred_res['classes'][0], pred_res['probabilities'][0])
            else:
                print(pred_res['label'][0], pred_res['probabilities'][0])

    else:

        model.train(
            input_fn=lambda: hys_input_fn_from_tfrecords(
                data_file=ARGS['train_data_tfrecords_dir'],
                num_epochs=ARGS['train_epoches_num'],
                shuffle=True,
                batch_size=ARGS['batch_size']
            )
        )
        results = model.evaluate(
            input_fn=lambda: hys_input_fn_from_tfrecords(
                data_file=ARGS['test_data_tfrecords_dir'],
                num_epochs=1,
                shuffle=False,
                batch_size=ARGS['batch_size']
            )
        )
        for key in sorted(results):
            print('%s: %s' % (key, results[key]))

        predictions = model.predict(
            input_fn=lambda: hys_input_fn_from_tfrecords(data_file=ARGS['test_data_tfrecords_dir'], num_epochs=1,
                                                         shuffle=False, batch_size=ARGS['batch_size'])
        )
        # for x in predictions:
        #     print(x['probabilities'][0])
        #     print(x['label'][0]))

        columns = ARGS['wide_columns'] + ARGS['deep_columns']+ ARGS['text_columns'] + ARGS['emb_columns']
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=columns)
        print('feature_spec:{0}'.format(feature_spec))
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel(EXPORT_PATH, serving_input_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(1)
    print(tf.__version__)
    parser = argparse.ArgumentParser(description='命令行中传入一个数字')
    # type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument('-f', "--file", type=str, help='file')
    args = parser.parse_args()
    f = open(args.file)
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    if 'census' in args.file:
        train_census_data(config)
    elif 'hys' in args.file:
        train_hys_data(config)
