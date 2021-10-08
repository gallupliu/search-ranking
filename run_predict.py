from __future__ import print_function
import grpc
import requests
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

path = '/Users/gallup/study/search-ranking/data/adult/test.csv'


# 生成tf.Example 数据
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


feature_dict = {}
serialized_strings = []
with open(path, encoding='utf-8') as f:
    lines = f.readlines()

    # names = [key for key in lines[0].strip('\n').split(',')]
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                'income_bracket', 'text']
    EMBEDDING_FEATURE_NAMES = ['bert_emb']
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
    TEXT_FEATURE_NAMES = ['text']
    CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_WITH_VOCABULARY.keys()) + list(
        CATEGORICAL_FEATURE_WITH_HASH_BUCKETS.keys())

    # {"age":[46.], "education_num":[10.],"capital_gain":[7688.], "capital_loss":[0.], "hours_per_week":[38.]}, {"age":[24.], "education_num":[13.],"capital_gain":[0.], "capital_loss":[0.], "hours_per_week":[50.],"text":[1,2,3,4,0]}]

    for i in range(len(lines)):
        items = [key for key in lines[i].strip('\n').split(',')]
        # print(len(items),len(names))
        # print('items:{0}'.format(items))
        # print('names:{0}'.format(names))
        for j,name in enumerate(names):
            item = items[j]
            print('names:{0} type:{1}'.format(name,type(name)))
            if name == 'label':
                continue
            if name in NUMERIC_FEATURE_NAMES:
                feature_dict[name] = _float_feature(float(item))
            elif name in CATEGORICAL_FEATURE_NAMES:
                feature_dict[name] = _bytes_feature(bytes(item, encoding='utf-8'))

            elif name in EMBEDDING_FEATURE_NAMES:
                feature_dict[name] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(value) for value in item.replace('[', '').replace(']', '').split(',')]))


            elif name in TEXT_FEATURE_NAMES:

                ids = [bytes(x, 'utf-8') for x in item.split(' ')]
                feature_dict[name] = _bytes_feature(bytes(item, encoding='utf-8'))
                # tokens_list = tf.strings.split(['1 2 4 6 0'], ' ')
                # ids = vocab.lookup(tokens_list)
                # print('ids:{0}'.format(ids))


        # feature_dict['text'] = _bytes_feature(bytes('3 9 4 8 0', encoding='utf-8'))
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        serialized = example_proto.SerializeToString()
        serialized_strings.append(serialized)

    # print(names)
    # print(types)
    # print(serialized_strings[0])
    #
    # example_proto = tf.train.Example.FromString(serialized_strings[0])
    # print(example_proto)

channel = grpc.insecure_channel(target='localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'deepfm'
request.model_spec.signature_name = 'serving_default'

data = serialized_strings
size = len(data)
request.inputs['examples'].CopyFrom(tf.make_tensor_proto(data, shape=[size]))

result = stub.Predict(request, 10.0)  # 10 secs timeout
print(result)