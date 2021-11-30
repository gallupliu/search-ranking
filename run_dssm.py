# -*- coding: utf-8 -*-
# @Time    : 2021/7/10 上午8:18
# @Author  : gallup
# @File    : run_dssm.py
import argparse
import os
import shutil
import yaml
import logging
import numpy as np
import tensorflow.compat.v1 as tf
from config.feature.match.census_match_feat_config_v1 import build_census_feat_columns
from models.wdl_v1 import wdl_estimator
from models.dssm_v2 import dssm_model_fn
from models.keras.models.dssm import DSSM
from tensorflow import keras
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from config.feature.match.hys_match_feat_config import CONFIG
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


class MyModule(tf.Module):
    def __init__(self, model, other_variable):
        self.model = model
        self._other_variable = other_variable

    @tf.function(input_signature=[{'keyword': tf.TensorSpec(shape=(None, 5), dtype=tf.int64)},
                                  {'item': tf.TensorSpec(shape=(None, 15), dtype=tf.int64)},
                                  {'type': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)},
                                  {'volume': tf.TensorSpec(shape=(None, 1), dtype=tf.string)},
                                  {'price': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)}]
                 )
    def score(self, keyword, item, volume, type, price):
        result = self.model.input
        return {"scores": self.model.output}

    # @tf.function(input_signature=[])
    # def metadata(self):
    #   return { "other_variable": self._other_variable }


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
        # 解析顺序乱了，重新定义顺序
        new_features = {}
        for feat in CONFIG['user_cols']:
            new_features[feat['name']] = features.pop(feat['name'])
        for feat in CONFIG['item_cols']:
            new_features[feat['name']] = features.pop(feat['name'])
        # new_features["item"] = features.pop("item")
        # new_features["type"] = features.pop("type")
        # new_features["volume"] = features.pop("volume")
        # new_features['price'] = features.pop('price')
        print('feature:{}'.format(features))
        print('new_feature:{}'.format(new_features))
        labels = features.pop('label')
        labels = tf.compat.v1.to_float(labels)
        return new_features, labels

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
def export_model(model):
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    # Set the path where the model will be saved.
    export_base_path = os.path.abspath('models/versions/')
    model_version = '1'
    export_path = os.path.join(tf.compat.as_bytes(export_base_path),
                               tf.compat.as_bytes(model_version))
    # Make the model builder.
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # Define the TensorInfo protocol buffer objects that encapsulate our
    # input/output tensors.
    # Note you can have a list of model.input layers, or just a single model.input
    # without any indexing. I'm showing a list of inputs and a single output layer.
    # Input tensor info.
    tensor_info_input0 = tf.saved_model.utils.build_tensor_info(model.input[0])
    tensor_info_input1 = tf.saved_model.utils.build_tensor_info(model.input[1])
    # Output tensor info.
    tensor_info_output = tf.saved_model.utils.build_tensor_info(model.output)

    # Define the call signatures used by the TF Predict API. Note the name
    # strings here should match what the layers are called in your model definition.
    # Might have to play with that because I forget if it's the name parameter, or
    # the actual object handle in your code.
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input0': tensor_info_input0, 'input1': tensor_info_input1},
            outputs={'prediction': tensor_info_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # Now we build the SavedModel protocol buffer object and then save it.
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={'predict': prediction_signature})
    builder.save(as_text=True)
def train_hys_data(config):
    # CONFIG["categorical_cols"]["UserID_idx"] = num_users
    # CONFIG["categorical_cols"]["Gender_idx"] = num_genders
    # CONFIG["categorical_cols"]["Age_idx"] = num_ages
    # CONFIG["categorical_cols"]["Occupation_idx"] = num_occupations
    # CONFIG["categorical_cols"]["MovieID_idx"] = num_movies
    # CONFIG["categorical_cols"]["Genres_idx"] = num_genres
    CONFIG["model_name"] = "dssm"
    CONFIG["vocab_size"] = 25
    CONFIG["num_sampled"] = 1
    CONFIG["l2_reg_embedding"] = 1e-6
    CONFIG["embed_size"] = 50

    train_ds = hys_input_fn_from_tfrecords("./match_feature.tfrecord/*.tfrecord", 1, shuffle=True, batch_size=4)
    print(train_ds)
    for x in train_ds.take(1):
        print('x:{0}'.format(x))
        # print(y)
    print('end ds')
    # model = build_mlp_model(CONFIG)
    model = DSSM(CONFIG)
    model = model.summary()
    model.summary()
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.RMSprop(), metrics=[AUC()])

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/logs_" + TIMESTAMP)
    # history = model.fit(train_ds,
    #                     epochs=5, steps_per_epoch=30,
    #                     callbacks=[tensorboard_callback]
    #                     )
    #
    model_file_path = './pb/match/' + CONFIG["model_name"] + '/'
    # # Calling `save('my_model')` creates a SavedModel folder `my_model`.
    # model.save(model_file_path)

    # module = MyModule(model, 1234)
    # tf.saved_model.save(module, model_file_path, signatures={"score": module.score})
    model_file_path_test = './pb/match/' + CONFIG["model_name"] + '/test/'
    @tf.function
    def serve(*args, **kwargs):
        outputs = model(*args, **kwargs)
        # Apply postprocessing steps, or add additional outputs.
        ...
        return outputs

    # arg_specs is `[tf.TensorSpec(...), ...]`. kwarg_specs, in this example, is
    # an empty dict since functional models do not use keyword arguments.
    arg_specs, kwarg_specs = model.save_spec()

    model.save(model_file_path_test , signatures={
        'serving_default': serve.get_concrete_function(*arg_specs, **kwarg_specs)
    })
    # import tempfile
    # model_dir = tempfile.mkdtemp()
    # # export_outputs = {
    # #                   'score': tf.estimator.export.ClassificationOutput}
    # keras_estimator = tf.keras.estimator.model_to_estimator(
    #     keras_model=model, model_dir=model_dir)
    # # estimator_train_result = keras_estimator.train(input_fn=lambda: input_fn(train_images, train_labels, EPOCHS, BATCH_SIZE))
    # keras_estimator.train(input_fn=lambda:hys_input_fn_from_tfrecords("./match_feature.tfrecord/*.tfrecord", 1, shuffle=True, batch_size=4), steps=500)
    # eval_result = keras_estimator.evaluate(input_fn=lambda:hys_input_fn_from_tfrecords("./match_feature.tfrecord/*.tfrecord", 1, shuffle=False, batch_size=4), steps=10)
    # print('Eval result: {}'.format(eval_result))


    test_input = {
        "keyword": np.asarray([[11, 4, 0, 0, 0],
                               [11, 4, 0, 0, 0],
                               [10, 19, 23, 0, 0],
                               [15, 21, 24, 0, 0]
                               ]),
        "item": np.asarray([[11, 4, 11, 4, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [11, 4, 11, 4, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [10, 19, 23, 10, 19, 23, 16, 24, 0, 0, 0, 0, 0, 0, 0],
                            [15, 21, 24, 15, 21, 24, 15, 21, 24, 0, 0, 0, 0, 0, 0]
                            ]),
        "volume": np.asarray([[0.2],
                              [0.2], [0.1], [0.3]
                              ]),  # [0,1)之间数据
        "type": np.asarray([['0'],
                            ['0'], ['0'], ['1']
                            ]),
        "price": np.asarray([[30.],
                             [30.], [10.], [19.]
                             ]),
    }

    # 输出节点名字要和pb模型里对应
    # test_target = {"output_1": np.asarray([[1.], [1.], [0.], [0.]])}
    test_target = {"tf.reshape_2": np.asarray([[1.], [1.], [0.], [0.]])}
    # 由模型生成预测
    def predict_input_fn(features, batch_size=256):
        """An input function for prediction."""
        # 将输入转换为无标签数据集。
        return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # predictions = keras_estimator.predict(
    #     input_fn=lambda: predict_input_fn(test_input))

    # columns = ARGS['user_columns'] + ARGS['item_columns']
    # feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=columns)
    # print('feature_spec:{0}'.format(feature_spec))
    # serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    # model.export_savedmodel(EXPORT_PATH, serving_input_fn)

    def serving_input_fn():

        feature_spec = {
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
        logging.debug("feature spec: %s", feature_spec)
        return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()
    #
    # keras_estimator.export_saved_model(model_file_path_test, serving_input_fn)
    # for pred_dict, expec in zip(predictions, test_target):
    #     class_id = pred_dict['class_ids'][0]
        # probability = pred_dict['probabilities'][class_id]
        #
        # print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        #     SPECIES[class_id], 100 * probability, expec))

    # It can be used to reconstruct the model identically.
    reconstructed_model = keras.models.load_model(model_file_path)

    # Let's check:

    # raw_predict_label = model.predict(test_input, callbacks=[tensorboard_callback])
    # reconstructed_predict_label = reconstructed_model.predict(test_input, callbacks=[tensorboard_callback])
    # print('raw:{0}'.format(raw_predict_label))
    # print('reconstructed:{0}'.format(reconstructed_predict_label))
    # # print("input shape:", test_input["item"].shape)
    # # print("predictions shape:", raw_predict_label.shape)
    # print(np.testing.assert_allclose(
    #     raw_predict_label,
    #     reconstructed_predict_label
    # ))

    # The reconstructed model is already compiled and has retrained the optimizer
    # state, so training can resume:
    # print('retrained')
    reconstructed_model.fit(test_input, test_target)

    # todo 1、解决输入batch 输出只有一个的问题 2、user侧和item分别导出模型 3、添加transformer和attention fusion 4、实现java的serving和local调用

    # user_embed_model = Model(inputs=model.user_input, outputs=model.user_embeding)
    # item_embed_model = Model(inputs=model.item_input, outputs=model.item_embeding)
    #
    # test_user_input = {
    #     "keyword": np.asarray([[11, 4, 0, 0, 0],
    #                            [11, 4, 0, 0, 0],
    #                            [10, 19, 23, 0, 0],
    #                            [15, 21, 24, 0, 0]
    #                            ]),
    # }
    #
    # test_item_input = {
    #     "item": np.asarray([[11, 4, 11, 4, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                         [11, 4, 11, 4, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                         [10, 19, 23, 10, 19, 23, 16, 24, 0, 0, 0, 0, 0, 0, 0],
    #                         [15, 21, 24, 15, 21, 24, 15, 21, 24, 0, 0, 0, 0, 0, 0]
    #                         ]),
    #     "volume": np.asarray([[0.2],
    #                           [0.2], [0.1], [0.3]
    #                           ]),  # [0,1)之间数据
    #     "type": np.asarray([['0'],
    #                         ['0'], ['0'], ['1']
    #                         ]),
    #     "price": np.asarray([[30.],
    #                          [30.], [10.], [19.]
    #                          ]),
    # }
    # user_embs = user_embed_model.predict(test_user_input )
    #
    # item_embs = item_embed_model.predict(test_item_input)
    #
    # user_embs = tf.squeeze(user_embs)
    # item_embs = tf.squeeze(item_embs)
    # print("user embed:{0}".format(user_embs))


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
