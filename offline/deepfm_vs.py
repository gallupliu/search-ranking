import datetime
import os

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

# 读取训练数据
from tensorflow.python.framework import graph_util
from tensorflow.keras.optimizers import Adam

print(tf.__version__)
df = pd.read_csv("./train_deepfm.csv")
labels = pd.read_csv("./train_deepfm_label.csv")

print(df.info())
print(labels.info())

features = df.to_dict('list')
labels = labels

# test_features = df[split_point+1:].to_dict('list')
# test_labels = labels[split_point+1:]
# 对不同的特征列做不同的处理
numeric_columns = ['age', 'fnlwgt',
                   'education_num',
                   'capital_gain',
                   'capital_loss',
                   'hours_per_week']
categorical_columns = ['education', 'marital_status', 'relationship', 'workclass', 'occupation', 'native_country',
                       'race', 'gender']
# fm 一阶部分需要one-hot类型的特征,r如果是连续类型的特征，可以直接输入到一阶部分；如果是离散类型的，可以做one-hot编码后再输入模型
# 所以，不管是连续类型的还是离散类型的，都可以做one-hot编码

# 为方便对离散特征做one-hot编码，我们需要构造离散特征做one-hot编码用到的 feature2int 字典
# 够造categorical特征做embedding or indicator用到的特征字典
categorical_columns_dict = {}
for col in categorical_columns:
    feature_list = df.groupby([col]).size().sort_values(ascending=False)
    categorical_columns_dict[col] = feature_list.index.tolist()


_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'

age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,gender,capital_gain,capital_loss,hours_per_week,native_country
]


_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], [0]]

# 构造feature_column
fm_first_order_fc = []
for col in numeric_columns:
    fc = tf.feature_column.numeric_column(col)
    fm_first_order_fc.append(fc)
for col in categorical_columns:
    fc = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            col,
            categorical_columns_dict[col]
        )
    )
    fm_first_order_fc.append(fc)
# print(fm_first_order_fc)

# 由于deepFM不是预定义模型，所以我们不能直接把feature_column作为参数输入到模型里，
# 必须通过input_layer接口转化成tensor，构造出feature_column对应的训练样本
fm_first_order_ip = tf.compat.v1.feature_column.input_layer(features, fm_first_order_fc)
print(fm_first_order_ip[0:2])
# fm_first_order_ip.shape=[1000. 1095],这表明连续特征+离散特征one-hot编码后，拼接在一块，一共有1095维度的特征
# 一阶部分的输入数据构造完成后，需要构造fm二阶部分和deep部分需要的embedding输入,二者公用一份embedding数据
# 连续特征的embedding和离散型数据的embedding有一些区别,连续性特征做embedding后需要乘以自身的值，离散型one-hot编码本身默认是1,embedding后
# 不要再乘以自身的值

# 连续特征先做Numeric转化，然后通过一个Dense网络来embedding
fm_second_order_numric_fc = []
for col in numeric_columns:
    fc = tf.feature_column.numeric_column(col)
    fm_second_order_numric_fc.append(fc)

embedding_dim = 16
fm_second_order_cate_fc = []
for col in categorical_columns:
    fc = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_list(col
                                                                  , categorical_columns_dict[col])
        , embedding_dim
    )
    fm_second_order_cate_fc.append(fc)

# 通过input_layer转化成 tensor
fm_second_order_numric_ip = tf.compat.v1.feature_column.input_layer(features, fm_second_order_numric_fc)
# fm_second_order_numric_ip = tf.reshape(fm_second_order_numric_ip, (-1, 1, len(numeric_columns)))
fm_second_order_cate_ip = tf.compat.v1.feature_column.input_layer(features, fm_second_order_cate_fc)
# 构造deepfm模型
# 一阶部分
fm_first_order_dim = fm_first_order_ip.shape[1].value
fm_first_order_input = tf.keras.Input(shape=(fm_first_order_dim), name='fm_first_oder_Input')
fm_first_order_output = tf.keras.layers.Dense(1, activation=tf.nn.relu, name='fm_first_oder_Output')(
    fm_first_order_input)
print(fm_first_order_input)
print(fm_first_order_output)
# 由于fm二阶部分和deep部分都需要并且共用embedding部分，我们先定义embedding部分
# numeric的embedding
embedding_numeric_Input = tf.keras.Input(shape=(1, len(numeric_columns)), name='embedding_numeric_Input')
print(embedding_numeric_Input)
input_tmp = []
for sli in range(embedding_numeric_Input.shape[2]):  # 输入的连续特征中每一个特征通过有个Dense网络得到一个dim维度的嵌入表示
    tmp_tensor = embedding_numeric_Input[:, :, sli]
    name = ("sli_%d" % sli)
    tmp_tensor = tf.keras.layers.Dense(embedding_dim, activation=tf.nn.relu, name=name)(tmp_tensor)
    input_tmp.append(tmp_tensor)
embedding_numeric_Input_concatenate = tf.keras.layers.concatenate(input_tmp, axis=1,
                                                                  name='numeric_embedding_concatenate')
print(embedding_numeric_Input_concatenate)

embedding_cate_Input = tf.keras.Input(shape=(len(categorical_columns) * embedding_dim), name='cat_embedding_Input')
# embedding_cate_Input = tf.reshape(embedding_cate_Input,(-1, len(categorical_columns),embedding_dim))
print(embedding_cate_Input)

embedding_Input = tf.keras.layers.concatenate([embedding_numeric_Input_concatenate, embedding_cate_Input], 1)
print(embedding_Input)
embedding_Input = tf.reshape(embedding_Input, (-1, len(numeric_columns + categorical_columns), embedding_dim))

# print(fm_second_order_numric_ip[0:2])
# print(fm_second_order_cate_ip[0:2])

# fm二阶部分
sum_then_square = tf.square(tf.reduce_sum(embedding_Input, 1))
print(sum_then_square)

square_then_sum = tf.reduce_sum(tf.square(embedding_Input), 1)
print(square_then_sum)

fm_second_order_output = tf.subtract(sum_then_square, square_then_sum)
# deep部分
deep_input = tf.reshape(embedding_Input
                        , (-1, (len(numeric_columns) + len(categorical_columns)) * embedding_dim)
                        , name='deep_input'
                        )
print(deep_input)
deep_x1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name='deep_x1')(deep_input)
deep_output = tf.keras.layers.Dense(16, activation=tf.nn.relu, name='deep_x2')(deep_x1)
# 想要更多层，可以继续增加
print(deep_output)
# concat output
deep_fm_out = tf.concat([fm_first_order_output, fm_second_order_output, deep_output], axis=1, name='caoncate_output')
print(deep_fm_out)
deep_fm_out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name='final_output')(deep_fm_out)
print(deep_fm_out)
# 建立模型，注意这是一个多输入单数输出的模型
model = tf.keras.Model(inputs=[fm_first_order_input, embedding_numeric_Input, embedding_cate_Input],
                       outputs=deep_fm_out)
model.compile(Adam(lr=0.01, decay=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy', 'AUC'])
print(model.summary())
tf.keras.utils.plot_model(model, to_file='DeepFM_V2.png', show_shapes=True)


def input_fn(epochs, batch_size):
    # 构造dataset
    input_tensor_dict = {'fm_first_oder_Input': fm_first_order_ip
        , 'embedding_numeric_Input': fm_second_order_numric_ip
        , 'cat_embedding_Input': fm_second_order_cate_ip}
    input_tensor_lable = {'final_output': labels}
    ds = tf.data.Dataset.from_tensor_slices((input_tensor_dict, input_tensor_lable)).shuffle(buffer_size=5000).batch(
        batch_size).repeat(epochs)
    return ds

def input_fn_new(data_path, shuffle, num_epochs, batch_size):
    """Generate an input function for the Estimator."""

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        # classes = tf.equal(labels, '>50K')  # binary classification
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_path)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10)

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)


    return dataset


estimator = tf.keras.estimator.model_to_estimator(model)
estimator.train(
    input_fn=lambda:input_fn(epochs=40, batch_size=2), steps=2000)

columns = fm_first_order_fc + fm_second_order_numric_fc + fm_second_order_cate_fc
feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=columns)
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
estimator.export_savedmodel('./', serving_input_fn)
