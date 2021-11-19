import json

import numpy as np
import tensorflow as tf
import pandas as pd


def load_feature_json(filename):
    global feature_conf_list
    f = open(filename, 'r')
    feature_conf_list = json.load(f)
    return feature_conf_list


load_feature_json('./data/look_order_model.json')


def embedding_dim(dim):
    """empirical embedding dim"""
    return int(np.power(2, np.ceil(np.log(dim ** 0.25))))


def get_feature_col_type():
    l_col = []
    l_col_type = []
    dict_col = {}
    for d in feature_conf_list:
        if d['feature_column_function'] == 'crossed_column':
            continue
        name = d['name']
        l_col.append(name)
        col_type = d['type']
        if col_type == 'tf.int64':
            l_col_type.append(tf.int64)
            dict_col[name] = 'int64'
        elif col_type == 'tf.float64':
            l_col_type.append(tf.float64)
            dict_col[name] = 'float64'
        else:
            l_col_type.append(tf.string)
            dict_col[name] = 'str'
    return l_col, l_col_type, dict_col


def get_data(data_file):
    l_col, _, dtype_dict = get_feature_col_type()
    print('usercols', l_col)
    print('dtype_dict', dtype_dict)
    df_data = pd.read_csv(data_file, usecols=l_col)

    for key, value in dtype_dict.items():
        if value == 'str':
            df_data[key] = df_data[key].astype(str)
        elif value == 'int64':
            df_data[key] = df_data[key].fillna(0).astype(int)
        elif value == 'float64':
            df_data[key] = df_data[key].astype(float)
        else:
            df_data[key] = df_data[key].dropna()

    return df_data


def sparse_string_join(input_sp):
    """Concats each row of SparseTensor `input_sp` and outputs them as a 1-D string tensor."""
    # convert the `SparseTensor` to a dense `Tensor`
    dense_input = tf.sparse_to_dense(input_sp.indices, input_sp.dense_shape, input_sp.values, default_value='')
    # remove extra spaces.
    return tf.string_strip(dense_input)


def get_seq_feature():
    ll = []
    for d in feature_conf_list:
        if d['feature_column_function'] == 'seq':
            ll.append(d['name'])
    return ll
