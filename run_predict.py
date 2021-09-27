from __future__ import print_function
import grpc
import requests
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

path = '/Users/Zhao/Desktop/tmp/adult_tf.csv'


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

    names = [key for key in lines[0].strip('\n').split(',')]
    types = [key for key in lines[1].strip('\n').split(',')]

    for i in range(2, len(lines)):
        items = [key for key in lines[i].strip('\n').split(',')]
        for j in range(len(items)):
            item = items[j]
            if types[j] == 'int':
                item = int(item)
                feature_dict[names[j]] = _float_feature(item)
            elif types[j] == 'string':
                feature_dict[names[j]] = _bytes_feature(bytes(item, encoding='utf-8'))
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
request.model_spec.name = 'adult_export_model'
request.model_spec.signature_name = 'predict'

data = serialized_strings
size = len(data)
request.inputs['examples'].CopyFrom(tf.contrib.util.make_tensor_proto(data, shape=[size]))

result = stub.Predict(request, 10.0)  # 10 secs timeout
print(result)