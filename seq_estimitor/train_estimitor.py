import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.framework import dtypes

from deepctr.estimator.inputs import input_fn_pandas
import din_estimator
import pandas as pd
from tensorflow import feature_column as fc

my_feature_columns = []


def create_feature_columns():
    # user feature
    driver_age_class = fc.embedding_column(
        fc.categorical_column_with_identity("driver_age", num_buckets=7, default_value=0), 32)

    # item feature
    pax_age_class = fc.embedding_column(fc.categorical_column_with_identity("pax_age", num_buckets=7, default_value=0),
                                        32)

    pax_des = fc.categorical_column_with_hash_bucket("des_id", 10000)
    pax_des_embed = fc.embedding_column(pax_des, 32)

    # context feature
    pax_price = tf.feature_column.numeric_column('price_id', default_value=0.0)
    pax_price_splits = tf.feature_column.bucketized_column(
        pax_price, boundaries=[10 * 100, 20 * 100, 30 * 100, 40 * 100, 50 * 100, 60 * 100, 70 * 100, 80 * 100, 90 * 100,
                               100 * 100, 110 * 100, 120 * 100])
    pax_price_embed = fc.embedding_column(pax_price_splits, 32)

    seq_cols = ['hist_price_id', 'hist_des_id']
    # hist_price_seq_embed = fc.embedding_column(fc.categorical_column_with_vocabulary_file(
    #     key='hist_price_id',
    #     vocabulary_file='./map.txt',
    #     num_oov_buckets=0), 32)
    # hist_des_seq_embed = fc.embedding_column(
    #     fc.categorical_column_with_vocabulary_file(key='hist_des_id', vocabulary_file='./map.txt',
    #                                                default_value=0), dimension=32)

    hist_price_seq_embed = fc.numeric_column(key='hist_price_id', shape=(3,), default_value=[0.0] * 3, dtype=tf.float32)

    hist_des_seq_embed = fc.numeric_column(key='hist_des_id', shape=(3,), default_value=[0.0] * 3, dtype=tf.float32)

    global my_feature_columns
    my_feature_columns = [driver_age_class, pax_age_class, pax_des_embed, pax_price_embed, hist_price_seq_embed,
                          hist_des_seq_embed]
    return my_feature_columns


feature_columns = create_feature_columns()
print(feature_columns)


# feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
# other_feature_spec = {
#     "hist_price_seq": tf.FixedLenFeature([3], tf.int64),
#     "hist_des_seq": tf.FixedLenFeature([3], tf.int64)
# }
# feature_spec.update(other_feature_spec)
# print(feature_spec)

def get_list(col):
    col = col.strip("[")
    col = col.strip("]")
    return col.split(',')


train = pd.read_csv('din_data.csv')
train['hist_price_id'] = train.agg(lambda x: get_list(x['hist_price_id']), axis=1)
train['hist_des_id'] = train.agg(lambda x: get_list(x['hist_des_id']), axis=1)
test = pd.read_csv('din_data.csv')
test['hist_price_id'] = test.agg(lambda x: get_list(x['hist_price_id']), axis=1)
test['hist_des_id'] = test.agg(lambda x: get_list(x['hist_des_id']), axis=1)
features = ['driver_age', 'pax_age', 'des_id', 'price_id', 'hist_price_id', 'hist_des_id']
print(train.columns)
train_model_input = input_fn_pandas(train, features, 'label', shuffle=True)
test_model_input = input_fn_pandas(test, features, None, shuffle=False)

model = din_estimator.DINEstimator(feature_columns, ['price_id', 'des_id'])

model.train(train_model_input)
pred_ans_iter = model.predict(test_model_input)
pred_ans = list(map(lambda x: x['pred'], pred_ans_iter))
#
print("test LogLoss", round(log_loss(test['label'].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test['label'].values, pred_ans), 4))

feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
model.export_savedmodel('./', serving_input_receiver_fn)
