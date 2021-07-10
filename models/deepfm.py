import numpy as np
import pandas as pd
import datetime
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from utils.feature_column import build_input_features, build_linear_embedding_dict, input_from_feature_columns, \
    get_linear_logit, build_embedding_dict, combined_dnn_input
from utils.embedding import FMLayer


########################################################################
#################定义模型##############
########################################################################


def DeepFM(linear_feature_columns, fm_group_columns, dict_categorical, dnn_hidden_units=(128, 128),
           dnn_activation='relu', seed=1024):
    """Instantiates the DeepFM Network architecture.
    Args:
        linear_feature_columns: An iterable containing all the features used by linear part of the model.
        fm_group_columns: list, group_name of features that will be used to do feature interactions.
        dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
        seed: integer ,to use as random seed.
        dnn_activation: Activation function to use in DNN
    return: A Keras model instance.
    """

    feature_columns = linear_feature_columns + fm_group_columns
    features = build_input_features(feature_columns)

    inputs_list = list(features.values())

    # 构建 linear embedding_dict
    linear_embedding_dict = build_linear_embedding_dict(linear_feature_columns)
    linear_sparse_embedding_list, linear_dense_value_list = input_from_feature_columns(features, feature_columns,
                                                                                       linear_embedding_dict,
                                                                                       dict_categorical)
    # linear part
    linear_logit = get_linear_logit(linear_sparse_embedding_list, linear_dense_value_list)

    # 构建 embedding_dict
    cross_columns = fm_group_columns
    embedding_dict = build_embedding_dict(cross_columns)
    sparse_embedding_list, _ = input_from_feature_columns(features, cross_columns, embedding_dict, dict_categorical)

    # 将所有sparse的k维embedding拼接起来，得到 (n, k)的矩阵，其中n为特征数，
    concat_sparse_kd_embed = Concatenate(axis=1, name="fm_concatenate")(sparse_embedding_list)  # ?, n, k
    # FM cross part
    fm_cross_logit = FMLayer()(concat_sparse_kd_embed)

    # DNN part
    dnn_input = combined_dnn_input(sparse_embedding_list, [])
    for i in range(len(dnn_hidden_units)):
        if i == len(dnn_hidden_units) - 1:
            dnn_out = Dense(units=dnn_hidden_units[i], activation='relu', name='dnn_' + str(i))(dnn_input)
            break
        dnn_input = Dense(units=dnn_hidden_units[i], activation='relu', name='dnn_' + str(i))(dnn_input)
    dnn_logit = Dense(1, use_bias=False, activation=None, kernel_initializer=tf.keras.initializers.glorot_normal(seed),
                      name='dnn_logit')(dnn_out)

    final_logit = Add()([linear_logit, fm_cross_logit, dnn_logit])

    output = tf.keras.layers.Activation("sigmoid", name="dfm_out")(final_logit)
    model = Model(inputs=inputs_list, outputs=output)

    return model
