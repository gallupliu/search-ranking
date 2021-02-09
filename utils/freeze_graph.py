import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python import pywrap_tensorflow


def get_tensor(checkpoint_path):
    # checkpoint_path = 'model.ckpt-xxx'
    # checkpoint_path = './uncased_L-12_H-768_A-12/bert_model.ckpt'
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)


def freeze_graph(ckpt, output_graph):
    output_node_names = 'bert/encoder/layer_10/output/dense/kernel'
    # saver = tf.train.import_meta_graph(ckpt+'.meta', clear_devices=True)
    saver = tf.compat.v1.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(',')
        )
        with tf.gfile.GFile(output_graph, 'wb') as fw:
            fw.write(output_graph_def.SerializeToString())
        print('{} ops in the final graph.'.format(len(output_graph_def.node)))



if __name__ == '__main__':
    checkpoint_path = '/Users/gallup/study/SearchPlatform/src/main/resources/bert/bert_model.ckpt'
    get_tensor(checkpoint_path)
    pb = '/Users/gallup/study/SearchPlatform/src/main/resources/bert/bert_model.pb'

    freeze_graph(checkpoint_path, pb)
