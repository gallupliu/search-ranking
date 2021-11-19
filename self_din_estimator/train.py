import tensorflow as tf

from deepctr.estimator.inputs import input_fn_pandas
from din_estimator import build_estimator, save_model, input_fn_v5
from utils import embedding_dim, get_data, load_feature_json
import pandas as pd
import feature_din

feature_conf_list = load_feature_json('./data/look_order_model.json')
from feature_din import context_cat_col, context_dense_col, driver_profile_cat_col, driver_profile_dense_col, \
    psg_profile_cat_col, psg_profile_dense_col

C_COLUMNS = feature_din.context_cat_col + feature_din.driver_profile_cat_col + feature_din.psg_profile_cat_col
D_COLUMNS = feature_din.context_dense_col + feature_din.driver_profile_dense_col + feature_din.psg_profile_dense_col
feature_name = C_COLUMNS + D_COLUMNS + ['same_city_look_order_price', 'same_city_look_order_start_distance',
                                        'same_city_look_order_time_diff']


def get_wide_deep_columns_new(data_file):
    df = get_data(data_file)
    print('len(df.columns)', len(df.columns))
    deep_columns = []
    wide_columns = []
    cross_dict = {}
    for feature_conf_info in feature_conf_list:
        feature_function = feature_conf_info['feature_column_function']
        feature_name = feature_conf_info['name']
        ###########################category############################
        if feature_function == 'categorical_column_with_vocabulary_list':
            voc_list = feature_conf_info.get('voc_list')
            if voc_list is None:
                voc_list = df[feature_name].unique()
            feature = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, voc_list)
            # feature = tf.feature_column.categorical_column_with_identity(feature_name, len(voc_list))
            wide_columns.append(feature)
            deep_columns.append(tf.feature_column.embedding_column(feature, dimension=8))

        elif feature_function == 'categorical_column_with_hash_bucket':  # 字符串映射，不想维护映射关系，hash_bucket_size=类别数2*5倍，会有hash冲突的问题
            hash_bucket_size = feature_conf_info['hash_bucket_size']
            feature = tf.feature_column.categorical_column_with_hash_bucket(feature_name,
                                                                            hash_bucket_size=hash_bucket_size)
            embed_dim = embedding_dim(hash_bucket_size)
            wide_columns.append(feature)
            deep_columns.append(tf.feature_column.embedding_column(feature, dimension=embed_dim))
        ###########################category#################################

        ###########################dense columns############################
        elif feature_function == 'numeric_column':
            deep_columns.append(tf.feature_column.numeric_column(feature_name))
        ###########################dense columns############################

        ###########################cross columns############################
        elif feature_name == 'same_city_look_order_start_distance':
            bucket_num = feature_conf_info['bucket_num']
            dim_value = bucket_num
            score_span = (df[feature_name].max() - df[feature_name].min()) // bucket_num + 1
            min_value = df[feature_name].min()
            same_city_look_order_start_distance_buckets = tf.feature_column.bucketized_column(
                tf.feature_column.numeric_column(feature_name),
                [(i * score_span + min_value) for i in
                 range(bucket_num)])
            wide_columns.append(same_city_look_order_start_distance_buckets)
            embed_dim = embedding_dim(dim_value)
            deep_columns.append(
                tf.feature_column.embedding_column(same_city_look_order_start_distance_buckets, dimension=embed_dim))
            cross_dict.update({'same_city_look_order_start_distance': same_city_look_order_start_distance_buckets})
        elif feature_name == 'same_city_look_order_price':
            bucket_num = feature_conf_info['bucket_num']
            dim_value = bucket_num
            score_span = (df[feature_name].max() - df[feature_name].min()) // bucket_num + 1
            min_value = df[feature_name].min()
            same_city_look_order_price_buckets = tf.feature_column.bucketized_column(
                tf.feature_column.numeric_column(feature_name),
                [(i * score_span + min_value) for i in
                 range(bucket_num)])
            wide_columns.append(same_city_look_order_price_buckets)
            embed_dim = embedding_dim(dim_value)
            deep_columns.append(
                tf.feature_column.embedding_column(same_city_look_order_price_buckets, dimension=embed_dim))
            cross_dict.update({'same_city_look_order_price': same_city_look_order_price_buckets})
        elif feature_name == 'same_city_look_order_time_diff':
            bucket_num = feature_conf_info['bucket_num']
            dim_value = bucket_num
            score_span = (df[feature_name].max() - df[feature_name].min()) // bucket_num + 1
            min_value = df[feature_name].min()
            same_city_look_order_time_diff_buckets = tf.feature_column.bucketized_column(
                tf.feature_column.numeric_column(feature_name),
                [(i * score_span + min_value) for i in
                 range(bucket_num)])
            wide_columns.append(same_city_look_order_time_diff_buckets)
            embed_dim = embedding_dim(dim_value)
            deep_columns.append(
                tf.feature_column.embedding_column(same_city_look_order_time_diff_buckets, dimension=embed_dim))
            cross_dict.update({'same_city_look_order_time_diff': same_city_look_order_time_diff_buckets})
        elif feature_function == 'crossed_column':  # 交叉特征的元素不能重复定义，所以只能单独拎出来
            start_distance_price_time_diff_feature = tf.feature_column.crossed_column(
                [cross_dict['same_city_look_order_start_distance'], cross_dict['same_city_look_order_price'],
                 cross_dict['same_city_look_order_time_diff']], hash_bucket_size=10000)
            wide_columns.append(start_distance_price_time_diff_feature)
            deep_columns.append(
                tf.feature_column.embedding_column(start_distance_price_time_diff_feature,
                                                   dimension=embedding_dim(10000)))

            start_distance_price_feature = tf.feature_column.crossed_column(
                [cross_dict['same_city_look_order_start_distance'], cross_dict['same_city_look_order_price']],
                hash_bucket_size=1000)
            wide_columns.append(start_distance_price_feature)
            deep_columns.append(tf.feature_column.embedding_column(start_distance_price_feature,
                                                                   dimension=embedding_dim(1000)))

            price_time_diff_feature = tf.feature_column.crossed_column(
                [cross_dict['same_city_look_order_price'], cross_dict['same_city_look_order_time_diff']],
                hash_bucket_size=1000)
            wide_columns.append(price_time_diff_feature)
            deep_columns.append(
                tf.feature_column.embedding_column(price_time_diff_feature,
                                                   dimension=embedding_dim(1000)))

            start_distance_time_diff_feature = tf.feature_column.crossed_column(
                [cross_dict['same_city_look_order_start_distance'], cross_dict['same_city_look_order_time_diff']],
                hash_bucket_size=1000)
            wide_columns.append(start_distance_time_diff_feature)
            deep_columns.append(
                tf.feature_column.embedding_column(start_distance_time_diff_feature,
                                                   dimension=embedding_dim(1000)))
        else:
            pass
        ###########################cross columns############################

    return wide_columns, deep_columns


def convert_int2str(col):
    return str(col)


def converType(hist_order_name, maxLen):
    if type(hist_order_name) == float:
        return ','.join(['0'] * maxLen)
    else:
        hist_order_name = hist_order_name.strip('[')
        hist_order_name = hist_order_name.strip(']')
        ans = []
        for elem in hist_order_name.split(','):
            ans.append(int(elem))
        if len(ans) < maxLen:
            ans = ans + [0] * (maxLen - len(ans))
    return ','.join(map(str, ans))


train_file = './data/input_train.csv'
test_file = './data/input_test.csv'
model_dir = './data/model'
export_dir = './data/export/'
wide_columns, deep_columns = get_wide_deep_columns_new(train_file)

print(deep_columns)
model = build_estimator(model_dir, deep_columns, hidden_units='256,128,64', learning_rate=0.0001, num_cross_layers=3,
                        feature_conf_list=feature_conf_list, mode='train')
print('model', model)
# model.train(
#     input_fn=lambda: input_fn(train_file, shuffle=True, num_epochs=40, batch_size=100), steps=2000)
feature_name = []
for d in feature_conf_list:
    if d['name'] == 'same_city_look_order_start_distance#same_city_look_order_price#same_city_look_order_time_diff':
        pass
    else:
        feature_name.append(d['name'])

print('feature_name', feature_name)
print('len of feature_name', len(feature_name))
# train = pd.read_csv(train_file)
# train['same_city_look_order_start_city_code'] = train.agg(
#     lambda x: convert_int2str(x['same_city_look_order_start_city_code']), axis=1)
# train['hist_order_name'] = train.agg(lambda x: converType(x['hist_order_name'], maxLen=50), axis=1)
#
# test = pd.read_csv(test_file)
# test['same_city_look_order_start_city_code'] = test.agg(
#     lambda x: convert_int2str(x['same_city_look_order_start_city_code']), axis=1)
# test['hist_order_name'] = test.agg(lambda x: converType(x['hist_order_name'], maxLen=50), axis=1)
#
# train = train[feature_name]
# test = test[feature_name]
# train.to_csv('./data/input_train.csv', index=0)
# test.to_csv('./data/input_test.csv', index=0)
model.train(input_fn=lambda: input_fn_v5(train_file, shuffle=False, num_epochs=1, batch_size=256),
            max_steps=2)
pred_dict = model.predict(
    input_fn=lambda: input_fn_v5(test_file, shuffle=False, num_epochs=1, batch_size=256))
pred_ans = list(map(lambda x: x['probabilities'], pred_dict))
print(len(pred_ans))

test_data = pd.read_csv(test_file)
from sklearn.metrics import log_loss, roc_auc_score

print("test LogLoss", round(log_loss(test_data['label'].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test_data['label'].values, pred_ans), 4))

save_model(model, deep_columns, feature_conf_list, export_dir)

# print('预测测试')
# examples = []
# test_data = pd.read_csv(test_file)
# inputs = test_data
# for index, row in inputs.iterrows():
#     feature = {}
#     for col, value in row.iteritems():
#         if isinstance(value, float):
#             feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#         elif isinstance(value, int):
#             feature[col] = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#         elif isinstance(value, str):
#             value_list = []
#             for elem in value.split(','):
#                 value_list.append(elem.encode())
#             print(value_list)
#             feature[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))
#     example = tf.train.Example(
#         features=tf.train.Features(
#             feature=feature
#         )
#     )
#     examples.append(example.SerializeToString())
# from tensorflow.contrib import predictor

# model_path = export_dir + '/1636688871'
# predict_fn = predictor.from_saved_model(model_path)
# print(predict_fn({'examples': examples}))
