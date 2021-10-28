import random
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.feature_column as fc

HYS_CONFIG = {
# label
    'columns': ['id','keyword', 'title', 'brand', 'tag', 'volume', 'type', 'user_bert_emb','item_bert_emb',
                'label'],
    'columns_defaults': [[-1], [''], [''],  [''], [''],  [0.0], [-1], [-1],
                         [-1]],
    'vocab_size': {
        'type': 3,

    },
    'vocab_file': './char.txt',
    'deep_emb_cols': [ 'type'],
    'deep_bucket_emb_cols': ['volume',],
    'wide_muti_hot_cols': ['type'],
    'wide_bucket_cols': ['volume'],
    'wide_cross_cols': [('type', 'volume'),],
    'text_cols': ['keyword', 'title', 'brand', 'tag'],
    'emb_cols': ['user_bert_emb','item_bert_emb'],
}


def build_hys_feat_columns(emb_dim=8):
    def _get_numeric_feat_range():
        # train = pd.read_csv('./data/raw/adult/adult.data', header=None, names=HYS_CONFIG['columns'])[
        #     HYS_CONFIG['deep_bucket_emb_cols']]
        # test = pd.read_csv('./data/raw/adult/adult.test', header=None, names=HYS_CONFIG['columns'])[
        #     HYS_CONFIG['deep_bucket_emb_cols']]
        # lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # train['text'] = train.apply(lambda x: ' '.join([str(x) for x in random.sample(lst, 4)]) + ' 0', axis=1)
        # test['text'] = test.apply(lambda x: ' '.join([str(x) for x in random.sample(lst, 4)]) + ' 0', axis=1)
        # train['bert_emb'] = train.apply(lambda x: np.random.uniform(low=-0.1, high=0.1, size=10).tolist(), axis=1)
        # test['bert_emb'] = test.apply(lambda x: np.random.uniform(low=-0.1, high=0.1, size=10).tolist(), axis=1)
        # total = pd.concat([train, test], axis=0)
        total = pd.read_csv('./hys_df_test.csv', header=None, names=HYS_CONFIG['columns'])[
            HYS_CONFIG['deep_bucket_emb_cols']]
        numeric_range = {}
        for col in HYS_CONFIG['deep_bucket_emb_cols']:
            numeric_range[col] = (total[col].min(), total[col].max())
        return numeric_range

    def _build_hys_deep_columns(emb_dim=8, numeric_range=None):
        feature_columns = []
        for col in HYS_CONFIG['deep_emb_cols']:
            feature_columns.append(
                fc.embedding_column(fc.categorical_column_with_hash_bucket(col, hash_bucket_size=1000 if
                HYS_CONFIG['vocab_size'][col] <= 1000 else HYS_CONFIG['vocab_size'][col] + 10000),
                                    dimension=emb_dim)
            )
        for col in HYS_CONFIG['deep_bucket_emb_cols']:
            feature_columns.append(
                fc.embedding_column(fc.bucketized_column(fc.numeric_column(col), boundaries=list(
                    np.linspace(numeric_range[col][0], numeric_range[col][1], 1000))), dimension=emb_dim)
            )
        for col in HYS_CONFIG['text_cols']:
            text_column = fc.categorical_column_with_vocabulary_file(
                key=col,
                vocabulary_file='./char.txt',
                num_oov_buckets=0)
            feature_columns.append(fc.embedding_column(text_column, 10))
        for col in HYS_CONFIG['emb_cols']:
            fc.numeric_column(key=col, shape=(10,), default_value=[0.0] * 10, dtype=tf.float32)

        feat_field_size = len(feature_columns)
        return feature_columns, feat_field_size

    def _build_hys_wide_columns(numeric_range=None):
        base_columns, cross_columns = [], []
        for col in HYS_CONFIG['wide_muti_hot_cols']:
            base_columns.append(
                fc.indicator_column(fc.categorical_column_with_hash_bucket(col, hash_bucket_size=1000 if
                HYS_CONFIG['vocab_size'][col] <= 1000 else HYS_CONFIG['vocab_size'][col] + 10000))
            )
        for col in HYS_CONFIG['wide_bucket_cols']:
            base_columns.append(
                fc.bucketized_column(fc.numeric_column(col),
                                     boundaries=list(np.linspace(numeric_range[col][0], numeric_range[col][1], 1000)))
            )
        for col in HYS_CONFIG['wide_cross_cols']:
            cross_columns.append(
                fc.indicator_column(fc.crossed_column([col[0], col[1]], hash_bucket_size=10000))
            )
        feature_columns = base_columns + cross_columns
        feat_field_size = len(feature_columns)
        return feature_columns, feat_field_size

    numeric_range = _get_numeric_feat_range()
    deep_columns, deep_fields_size = _build_hys_deep_columns(emb_dim, numeric_range)
    wide_columns, wide_fields_size = _build_hys_wide_columns(numeric_range)
    feat_config = {
        'deep_columns': deep_columns,
        'deep_fields_size': deep_fields_size,
        'wide_columns': wide_columns,
        'wide_fields_size': wide_fields_size,
        'embedding_dim': emb_dim,
    }
    return feat_config


def official_hys_feature_columns_config_demo():
    # categorical_column
    gender = fc.categorical_column_with_vocabulary_list('gender', ['Female', 'Male'])
    education = fc.categorical_column_with_vocabulary_list('education',
                                                           ['Bachelors', 'HS-grad', '11th', 'Masters', '9th',
                                                            'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th',
                                                            'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th',
                                                            'Preschool', '12th'])
    marital_status = fc.categorical_column_with_vocabulary_list('marital_status', ['Married-civ-spouse', 'Divorced',
                                                                                   'Married-spouse-absent',
                                                                                   'Never-married', 'Separated',
                                                                                   'Married-AF-spouse', 'Widowed'])
    relationship = fc.categorical_column_with_vocabulary_list('relationship',
                                                              ['Husband', 'Not-in-family', 'Wife', 'Own-child',
                                                               'Unmarried', 'Other-relative'])
    workclass = fc.categorical_column_with_vocabulary_list('workclass',
                                                           ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
                                                            'Local-gov', '?', 'Self-emp-inc', 'Without-pay',
                                                            'Never-worked'])

    # To show an example of hashing:
    native_country = fc.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000)
    occupation = fc.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)

    # Continuous feature columns.
    age = fc.numeric_column('age')
    education_num = fc.numeric_column('education_num')
    capital_gain = fc.numeric_column('capital_gain')
    capital_loss = fc.numeric_column('capital_loss')
    hours_per_week = fc.numeric_column('hours_per_week')

    # bucketized transformations.
    age_buckets = fc.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    base_columns = [gender, education, marital_status, relationship, workclass, occupation, native_country, age_buckets]
    crossed_columns = [
        fc.crossed_column(['education', 'occupation'], hash_bucket_size=1000),
        fc.crossed_column([age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
        fc.crossed_column(['native_country', 'occupation'], hash_bucket_size=1000)
    ]
    feature_columns = [
        fc.indicator_column(workclass),
        fc.indicator_column(education),
        fc.indicator_column(gender),
        fc.indicator_column(relationship),
        fc.embedding_column(native_country, dimension=32),
        fc.embedding_column(occupation, dimension=32),
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
    ]
    return feature_columns, base_columns, crossed_columns
