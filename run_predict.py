
#  -*- coding: UTF-8 -*
from __future__ import print_function
import requests
import sys
import time
import threading

# This is a placeholder for a Google-internal import.

import grpc
import numpy
import tensorflow as tf
import datetime
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from utils.utils import fn_timer

class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(label, result_counter):
    """Creates RPC callback function.

    Args:
      label: The correct label for the predicted example.
      result_counter: Counter for the prediction result.
    Returns:
      The callback function.
    """

    def _callback(result_future):
        """Callback function.

        Calculates the statistics for the prediction result.

        Args:
          result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            response = numpy.array(
                result_future.result().outputs['out_y'].int64_val)

            for i in range(len(label)):
                if label[i] != response[i]:
                    result_counter.inc_error()
        result_counter.inc_done()
        result_counter.dec_active()

    return _callback

@fn_timer
def do_dssm_rest_inference(hostport, test_input):
    req_list = []
    data = {"instances": req_list}
    for i in range(len(test_input['item'])):
        instance = {}
        for k,v in test_input.items():
            instance[k] = v[i]
        req_list.append(instance)

    try:
        resp = requests.post('http://' + hostport + '/v1/models/dssm:predict', json=data)
        resp.raise_for_status()  # 如果响应状态码不是 200，就主动抛出异常
    except requests.RequestException as e:
        print(e)
    else:
        result = resp.json()
        print(type(result), result['predictions'], sep='\n')


@fn_timer
def do_dssm_grpc_inference(hostport, concurrency, test_input):
    """Tests PredictionService with concurrent requests.

    Args:
      hostport: Host:port address of the PredictionService.
      concurrency: Maximum number of concurrent requests.
      num_tests: Number of test images to use.

    Returns:
      The classification error rate.

    Raises:
      IOError: An error occurred processing test data set.
    """

    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'dssm'
    request.model_spec.signature_name = 'serving_default'

    request.inputs["item"].CopyFrom(
        tf.make_tensor_proto(np.asarray(test_input["item"]), shape=[4, len(test_input["item"][0])], dtype=tf.int32))

    request.inputs["keyword"].CopyFrom(
        tf.make_tensor_proto(np.asarray(test_input["keyword"]), shape=[4, len(test_input["keyword"][0])], dtype=tf.int32))
    request.inputs["volume"].CopyFrom(
        tf.make_tensor_proto(np.asarray(test_input["volume"]), shape=[4, len(test_input["volume"][0])], dtype=tf.float32))
    request.inputs["type"].CopyFrom(
        tf.make_tensor_proto(np.asarray(test_input["type"]), shape=[4, len(test_input["type"][0])], dtype=tf.string))
    request.inputs["price"].CopyFrom(
        tf.make_tensor_proto(np.asarray(test_input["price"]), shape=[4, len(test_input["price"][0])], dtype=tf.float32))

    # res = stub.Predict(request, 5)
    resp = stub.Predict.future(request, 5.0)  # 5 seconds
    res = resp.result().outputs
    print(res)




def main():
    #
    # test_input = {
    #   "item": np.asarray([[11, 4, 11, 4, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                       [11, 4, 11, 4, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                       [10, 19, 23, 10, 19, 23, 16, 24, 0, 0, 0, 0, 0, 0, 0],
    #                       [15, 21, 24, 15, 21, 24, 15, 21, 24, 0, 0, 0, 0, 0, 0]
    #                       ]),
    #   "keyword": np.asarray([[11, 4, 0, 0, 0],
    #                          [11, 4, 0, 0, 0],
    #                          [10, 19, 23, 0, 0],
    #                          [15, 21, 24, 0, 0]
    #                          ]),
    #
    #   "volume": np.asarray([[0.2],
    #                         [0.2], [0.1], [0.3]
    #                         ]),  # [0,1)之间数据
    #   "type": np.asarray([['0'],
    #                       ['0'], ['0'], ['1']
    #                       ]),
    #   "price": np.asarray([[30.],
    #                        [30.], [10.], [19.]
    #                        ]),
    # }

    test_input = {
        "item": [[11, 4, 11, 4, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [11, 4, 11, 4, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [10, 19, 23, 10, 19, 23, 16, 24, 0, 0, 0, 0, 0, 0, 0],
                 [15, 21, 24, 15, 21, 24, 15, 21, 24, 0, 0, 0, 0, 0, 0]
                 ],
        "keyword": [[11, 4, 0, 0, 0],
                    [11, 4, 0, 0, 0],
                    [10, 19, 23, 0, 0],
                    [15, 21, 24, 0, 0]
                    ],

        "volume": [[0.2],
                   [0.2], [0.1], [0.3]
                   ],  # [0,1)之间数据
        "type": [['0'],
                 ['0'], ['0'], ['1']
                 ],
        "price": [[30.],
                  [30.], [10.], [19.]
                  ],
    }

    start_time = datetime.datetime.now()
    do_dssm_grpc_inference("127.0.0.1:8500", 1, test_input)
    print("Cost {0} seconds to predict".
          format(datetime.datetime.now() - start_time))
    do_dssm_rest_inference("127.0.0.1:8501", test_input)


if __name__ == '__main__':
    main()
