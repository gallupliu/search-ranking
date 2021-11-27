# -*- coding: utf-8 -*-
# @Time    : 2021/7/10 上午8:18
# @Author  : gallup
# @File    : run_dssm.py
import argparse
import os
import shutil
import yaml
import tensorflow.compat.v1 as tf
from config.feature.match.census_match_feat_config_v1 import build_census_feat_columns
from models.wdl_v1 import wdl_estimator
from models.dssm_v2 import dssm_model_fn
from models.keras.models.dssm import DSSM
from tensorflow import keras
from config.feature.match.hys_match_feat_config import HYS_CONFIG
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
        'dssm': dssm_model_fn,
        # 'resnet':   res_model_fn,
        # 'pnn':      pnn_model_fn,
        # 'fibinet':  fibinet_model_fn,
        # 'afm':      afm_model_fn,
    }

    print(ARGS['model_name'])
    ARGS['ckpt_dir'] = CENSUS_PATH + 'ckpt_dir/' + ARGS['model_name']
    # ARGS['embedding_dim'] = feat_columns['embedding_dim']
    # ARGS['deep_columns'] = feat_columns['deep_columns']
    # ARGS['deep_fields_size'] = feat_columns['deep_fields_size']
    # ARGS['wide_columns'] = feat_columns['deep_columns']
    # ARGS['wide_fields_size'] = feat_columns['wide_fields_size']
    ARGS['user_columns'] = feat_columns['user_columns']
    ARGS['user_fields_size'] = feat_columns['user_fields_size']
    ARGS['item_columns'] = feat_columns['item_columns']
    ARGS['item_fields_size'] = feat_columns['item_fields_size']
    ARGS['model_fn_map'] = MODEL_FN_MAP
    print("emb_columns:{0},text_columns:{1}".format(feat_columns['user_columns'], feat_columns['item_columns']))
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
        input_fn=lambda: census_text_input_fn_from_tfrecords(data_file=ARGS['test_data_tfrecords_dir'], num_epochs=1,
                                                             shuffle=False, batch_size=ARGS['batch_size'])
    )
    # for x in predictions:
    #     print(x['probabilities'][0])
    #     print(x['label'][0]))

    columns = ARGS['user_columns'] + ARGS['item_columns']
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=columns)
    print('feature_spec:{0}'.format(feature_spec))
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    model.export_savedmodel(EXPORT_PATH, serving_input_fn)


def hys_input_fn_from_tfrecords(data_file, num_epochs, shuffle, batch_size):
    def _parse_func(record):
        features = {
            # int
            # "id": tf.io.FixedLenFeature([], tf.int64),
            # string
            "keyword": tf.io.FixedLenFeature([5], tf.int64),
            # "title": tf.io.FixedLenFeature([5], tf.string),
            # "brand": tf.io.FixedLenFeature([5], tf.string),
            # "tag": tf.io.FixedLenFeature([5], tf.string),
            "item": tf.io.FixedLenFeature([15], tf.int64),
            "type": tf.io.FixedLenFeature([], tf.string),

            "volume": tf.io.FixedLenFeature([], tf.float32),
            "price": tf.io.FixedLenFeature([], tf.float32),
            # 'user_bert_emb': tf.io.FixedLenFeature([10], tf.float32),  # query向量
            # 'item_bert_emb': tf.io.FixedLenFeature([10], tf.float32),  # item向量
            "label": tf.io.FixedLenFeature([], tf.int64),

        }
        features = tf.io.parse_single_example(record, features)
        labels = features.pop('label')
        labels = tf.compat.v1.to_float(labels)
        return features, labels

    # tf.compat.v1.gfile.Glob(path2)
    print(tf.io.gfile.listdir)
    print('data_file:{0}'.format(data_file))
    # assert tf.io.gfile.exists(tf.io.gfile.glob(data_file)), ('no file named: ' + str(data_file))
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(data_file)).map(_parse_func,
                                                                       num_parallel_calls=10)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    # iterator = dataset.make_one_shot_iterator()
    # print('ds:{0}'.format(dataset))
    # features, labels = iterator.get_next()
    # labels = tf.compat.v1.to_float(labels)
    # return features, labels
    return dataset


def train_hys_data(config):
    # HYS_CONFIG["categorical_cols"]["UserID_idx"] = num_users
    # HYS_CONFIG["categorical_cols"]["Gender_idx"] = num_genders
    # HYS_CONFIG["categorical_cols"]["Age_idx"] = num_ages
    # HYS_CONFIG["categorical_cols"]["Occupation_idx"] = num_occupations
    # HYS_CONFIG["categorical_cols"]["MovieID_idx"] = num_movies
    # HYS_CONFIG["categorical_cols"]["Genres_idx"] = num_genres
    HYS_CONFIG["vocab_size"] = 25
    HYS_CONFIG["num_sampled"] = 1
    HYS_CONFIG["l2_reg_embedding"] =1e-6
    HYS_CONFIG["embed_size"] = 50

    train_ds = hys_input_fn_from_tfrecords("./match_feature.tfrecord/*.tfrecord", 1, shuffle=True, batch_size=4)
    print(train_ds)
    for x in train_ds.take(1):
        print('x:{0}'.format(x))
        # print(y)
    print('end ds')
    # model = build_mlp_model(HYS_CONFIG)
    model = DSSM(HYS_CONFIG)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.RMSprop())

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/logs_" + TIMESTAMP)
    model.fit(train_ds,
              epochs=5, steps_per_epoch=30,
              callbacks=[tensorboard_callback]
              )

    # inputs = df.sample(frac=1.0)[
    #     ["UserID_idx", "Gender_idx", "Age_idx", "Occupation_idx", "MovieID_idx", "Genres_idx", "Title", "query"]].head(
    #     10)
    #
    # # 对于（用户ID，召回的电影ID列表），计算分数
    # model.predict([
    #     inputs["UserID_idx"],
    #     inputs["Gender_idx"],
    #     inputs["Age_idx"],
    #     inputs["Occupation_idx"],
    #     inputs["query"],
    #     inputs["MovieID_idx"],
    #     inputs["Genres_idx"],
    #     inputs["Title"],
    #
    # ])
    #
    # model.save("./data/ml-latest-small/tensorflow_two_tower.h5")
    #
    # new_model = tf.keras.models.load_model("./data/ml-latest-small/tensorflow_two_tower.h5")
    #
    # new_model.predict([
    #     inputs["UserID_idx"],
    #     inputs["Gender_idx"],
    #     inputs["Age_idx"],
    #     inputs["Occupation_idx"],
    #     inputs["query"],
    #     inputs["MovieID_idx"],
    #     inputs["Genres_idx"],
    #     inputs["Title"],
    # ])
    #
    # df_movie["Title_str"] = df_movie["Title"].map(lambda x: re.sub(pattern, '', x).lower().strip().split(" "))
    # df_user["query"] = df_movie["Title_str"].map(lambda x: x[:3])  # 暂取title前十个字符作为query
    # df_movie["Title_ids"] = df_movie["Title_str"].map(lambda x: encode_text(x))
    #
    # Title_ids = keras.preprocessing.sequence.pad_sequences(df_movie['Title_ids'], value=0, padding='post', maxlen=10)
    # # df_user["query_str"] = df_user["query"].map(lambda x: re.sub(pattern, '', x).lower().strip().split(" "))
    # df_user["query_ids"] = df_user["query"].map(lambda x: encode_text(x))
    # query_ids = keras.preprocessing.sequence.pad_sequences(df_user['query_ids'], value=0, padding='post', maxlen=10)
    #
    # movie_layer_model = keras.models.Model(
    #     inputs=[model.input[5], model.input[6], model.input[7]],
    #     outputs=model.get_layer("movie_embedding").output
    # )
    #
    # movie_embeddings = []
    # for index, row in df_movie.iterrows():
    #     movie_id = row["MovieID"]
    #     movie_input = [
    #         np.reshape(row["MovieID_idx"], [1, 1]),
    #         np.reshape(row["Genres_idx"], [1, 1]),
    #         np.reshape([111, 11, 1, 1, 1, 99, 89, 10, 0, 0], [1, 10]),
    #
    #     ]
    #     movie_embedding = movie_layer_model(movie_input)
    # #
    # #     embedding_str = ",".join([str(x) for x in movie_embedding.numpy().flatten()])
    # #     movie_embeddings.append([movie_id, embedding_str])
    # #
    # # df_movie_embedding = pd.DataFrame(movie_embeddings, columns=["movie_id", "movie_embedding"])
    # #
    # # output = "./data/ml-latest-small/tensorflow_movie_embedding.csv"
    # # df_movie_embedding.to_csv(output, index=False)
    #
    # user_layer_model = keras.models.Model(
    #     inputs=[model.input[0], model.input[1], model.input[2], model.input[3], model.input[4]],
    #     outputs=model.get_layer("user_embedding").output
    # )
    #
    # user_embeddings = []
    # for index, row in df_user.iterrows():
    #     user_id = row["UserID"]
    #     user_input = [
    #         np.reshape(row["UserID_idx"], [1, 1]),
    #         np.reshape(row["Gender_idx"], [1, 1]),
    #         np.reshape(row["Age_idx"], [1, 1]),
    #         np.reshape(row["Occupation_idx"], [1, 1]),
    #         np.reshape([111, 11, 1, 1, 1, 99, 89, 10, 0, 0], [1, 10]),
    #     ]
    #     user_embedding = user_layer_model(user_input)
    #
    # #     embedding_str = ",".join([str(x) for x in user_embedding.numpy().flatten()])
    # #     user_embeddings.append([user_id, embedding_str])
    # #
    # # df_user_embedding = pd.DataFrame(user_embeddings, columns=["user_id", "user_embedding"])
    # #
    # # output = "./data/ml-latest-small/tensorflow_user_embedding.csv"
    # # df_user_embedding.to_csv(output, index=False)


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
    else:
        train_hys_data(config)
