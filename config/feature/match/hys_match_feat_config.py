import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.feature_column as fc

FEAT_CONFIG = {
    # label
    'columns': ['id', 'keyword', "item", 'title', 'brand', 'tag', 'volume', 'type', 'price', 'user_bert_emb',
                'item_bert_emb',
                'label'],
    'vocab_size': {
        'type': 3,

    },
    'vocab_file': './char.txt',
    'deep_emb_cols': ['type'],
    'deep_bucket_emb_cols': ['volume', 'price'],
    'wide_muti_hot_cols': ['type'],
    'wide_bucket_cols': ['volume', 'price'],
    'wide_cross_cols': [('type', 'volume'), ],
    'text_cols': ['keyword', "item"],
    # 'bert_text_cols': ['keyword_input_ids', 'keyword_attention_mask', 'item_input_ids', 'item_attention_mask'],
    'emb_cols': ['user_bert_emb', 'item_bert_emb'],
    'categorical_cols': ['type'],  # 类别型特征统一为string格式
    'numeric_cols': ['volume'],  # 数值型，范围【0，1），直接输入进模型
    'bucket_cols': ['price'],  # 数值分桶型，必须分桶，也就是对应cols 必须有bins
    'crossed_cols': {},
    'user_cols': [{'name': 'keyword', 'num': 5, 'embed_dim': 50},
                  # {'name': 'keyword_input_ids', 'num': 10, },
                  # {'name': 'keyword_attention_mask', 'num': 10, }
                  ]
    ,
    'item_cols': [{'name': 'item', 'num': 15, 'embed_dim': 50},
                  # {'name': 'item_input_ids', 'num': 20, }, {'name': 'item_attention_mask', 'num': 20, },
                  {'name': 'type', 'num': 4, 'embed_dim': 2, 'vocab_list': ['0', '1', '2']},
                  {'name': 'volume', 'num': 1, 'embed_dim': 2},
                  {'name': 'price', 'num': 7, 'embed_dim': 2, 'bins': [0, 10, 20, 30, 40, 50]}
                  ]
}


class FeatureConfig(object):
    def __init__(self, df, config):
        # self.user_feature_columns = dict()
        # self.item_feature_columns = dict()
        # self.all_columns = dict()
        # self.feature_spec = dict()
        self.config = config
        self.user_feature_columns = []
        self.item_feature_columns = []
        self.numeric_range = self._get_numeric_feat_range(df)

        for key, value in self.user_feature_columns.items():
            self.all_columns[key] = value
        for key, value in self.item_feature_columns.items():
            self.all_columns[key] = value

        self.feature_spec = tf.feature_column.make_parse_example_spec(self.all_columns.values())

    def _get_numeric_feat_range(self, df):
        """
        @param df:
        @return:
        """
        total = df.select(*FEAT_CONFIG['bucket_cols']).toPandas()
        numeric_range = {}
        for col in FEAT_CONFIG['bucket_cols']:
            numeric_range[col] = (total[col].min(), total[col].max())
        return numeric_range

    def create_features_columns(self, cols):
        """

        @param cols:
        @return:
        """
        #
        #         k in self.categorical_cols:
        #         item_feature_inputs[k] = item_inputs[k]
        #         category = fc.categorical_column_with_vocabulary_list(
        #             k, feat['vocab_list'])
        #         category_column = fc.embedding_column(category, feat['embed_dim'])
        #         item_feature_columns.append(category_column)
        #
        #         # print('category_column:{0}'.format(category_column))
        #         # category_feature_layer = tf.keras.layers.DenseFeatures(category_column)
        #         # category_feature_outputs = category_feature_layer(item_inputs)
        #         # print('category_feature_outputs{0}'.format(category_feature_outputs))
        #
        #     elif k in self.numeric_cols:
        #     item_feature_inputs[k] = item_inputs[k]
        #     feat_col = fc.numeric_column(feat['name'])
        #     item_feature_columns.append(feat_col)
        #
        #     # print('feat_col:{0}'.format(feat_col))
        #     # feat_col_layer = tf.keras.layers.DenseFeatures(feat_col)
        #     # feat_col_outputs = feat_col_layer(item_inputs)
        #     # print('feat_col_outputs {0}'.format(feat_col_outputs ))
        #
        #
        # if k in self.bucket_cols:
        #     item_feature_inputs[k] = item_inputs[k]
        #     feat_buckets = fc.bucketized_column(feat_col, boundaries=feat['bins'])
        #     item_feature_columns.append(feat_buckets)
        #
        feature_columns = []
        for col in cols:
            if col['name'] in FEAT_CONFIG['text_cols']:
                # text_column = fc.categorical_column_with_vocabulary_file(
                #     key=col,
                #     vocabulary_file='./ids.txt',
                #     num_oov_buckets=0)
                text_column = fc.sequence_categorical_column_with_vocabulary_file(
                    key=col['name'], vocabulary_file=self.config['char_file'],
                    num_oov_buckets=5)
                # feature_columns.append(fc.embedding_column(text_column, 10))
                feature_columns.append(fc.shared_embeddings(text_column, self.config['embed_size']))
            if col['name'] in FEAT_CONFIG['categorical_cols']:
                category = fc.categorical_column_with_vocabulary_list(
                    col['name'], col['vocab_list'])
                category_column = fc.embedding_column(category, col['embed_dim'])
                feature_columns.append(category_column)

            if col['name'] in self.numeric_cols:
                feat_col = fc.numeric_column(col['name'])
                feature_columns.append(feat_col)

            if col['name'] in FEAT_CONFIG['bucket_cols']:
                feature_columns.append(
                    fc.embedding_column(fc.bucketized_column(fc.numeric_column(col), boundaries=list(
                        np.linspace(self.numeric_range[col['name']][0], self.numeric_range[col['name']][1], 100))),
                                        dimension=col['embed_dim'])
                )
            return feature_columns


def build_hys_feat_columns(emb_dim=8):
    def _get_numeric_feat_range():
        train = pd.read_csv('./data/raw/adult/adult.data', header=None, names=FEAT_CONFIG['columns'])[
            FEAT_CONFIG['bucket_cols']]
        test = pd.read_csv('./data/raw/adult/adult.test', header=None, names=FEAT_CONFIG['columns'])[
            FEAT_CONFIG['bucket_cols']]
        # lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # train['text'] = train.apply(lambda x: ' '.join([str(x) for x in random.sample(lst, 4)]) + ' 0', axis=1)
        # test['text'] = test.apply(lambda x: ' '.join([str(x) for x in random.sample(lst, 4)]) + ' 0', axis=1)
        # train['bert_emb'] = train.apply(lambda x: np.random.uniform(low=-0.1, high=0.1, size=10).tolist(), axis=1)
        # test['bert_emb'] = test.apply(lambda x: np.random.uniform(low=-0.1, high=0.1, size=10).tolist(), axis=1)
        total = pd.concat([train, test], axis=0)
        numeric_range = {}
        for col in FEAT_CONFIG['bucket_cols']:
            numeric_range[col] = (total[col].min(), total[col].max())
        return numeric_range

    def _build_hys_user_columns(numeric_range=None):
        feature_columns = []

        for col in FEAT_CONFIG['user_cols']:
            if col in FEAT_CONFIG['text_cols']:
                # text_column = fc.categorical_column_with_vocabulary_file(
                #     key=col,
                #     vocabulary_file='./ids.txt',
                #     num_oov_buckets=0)
                text_column = fc.sequence_categorical_column_with_vocabulary_file(
                    key=col, vocabulary_file='./char.txt',
                    num_oov_buckets=5)
                # feature_columns.append(fc.embedding_column(text_column, 10))
                feature_columns.append(fc.shared_embeddings(text_column, 10))

            if col in FEAT_CONFIG['emb_cols']:
                feature_columns.append(
                    fc.numeric_column(key=col, shape=(10,), default_value=[0.0] * 10, dtype=tf.float32))

            if col in FEAT_CONFIG['deep_emb_cols']:
                feature_columns.append(
                    fc.embedding_column(fc.categorical_column_with_hash_bucket(col, hash_bucket_size=10 if
                    FEAT_CONFIG['vocab_size'][col] <= 100 else FEAT_CONFIG['vocab_size'][col] + 100),
                                        dimension=emb_dim)
                )
            if col in FEAT_CONFIG['bucket_cols']:
                feature_columns.append(
                    fc.embedding_column(fc.bucketized_column(fc.numeric_column(col), boundaries=list(
                        np.linspace(numeric_range[col][0], numeric_range[col][1], 100))), dimension=emb_dim)
                )

        feat_field_size = len(feature_columns)
        return feature_columns, feat_field_size

    def _build_hys_item_columns(numeric_range=None):
        feature_columns = []

        for col in FEAT_CONFIG['item']:
            if col in FEAT_CONFIG['text_cols']:
                # text_column = fc.categorical_column_with_vocabulary_file(
                #     key=col,
                #     vocabulary_file='./ids.txt',
                #     num_oov_buckets=0)
                text_column = fc.sequence_categorical_column_with_vocabulary_file(
                    key=col, vocabulary_file='./char.txt',
                    num_oov_buckets=5)
                feature_columns.append(fc.embedding_column(text_column, 10))

            if col in FEAT_CONFIG['emb_cols']:
                feature_columns.append(
                    fc.numeric_column(key=col, shape=(10,), default_value=[0.0] * 10, dtype=tf.float32))

            if col in FEAT_CONFIG['deep_emb_cols']:
                feature_columns.append(
                    fc.embedding_column(fc.categorical_column_with_hash_bucket(col, hash_bucket_size=10 if
                    FEAT_CONFIG['vocab_size'][col] <= 100 else FEAT_CONFIG['vocab_size'][col] + 100),
                                        dimension=emb_dim)
                )
            if col in FEAT_CONFIG['bucket_cols']:
                feature_columns.append(
                    fc.embedding_column(fc.bucketized_column(fc.numeric_column(col), boundaries=list(
                        np.linspace(numeric_range[col][0], numeric_range[col][1], 100))), dimension=emb_dim)
                )
        feat_field_size = len(feature_columns)
        return feature_columns, feat_field_size

    numeric_range = _get_numeric_feat_range()
    user_columns, user_fields_size = _build_hys_user_columns(numeric_range)
    item_columns, item_fields_size = _build_hys_item_columns(numeric_range)

    feat_config = {
        'user_columns': user_columns,
        'user_fields_size': user_fields_size,
        'item_columns': item_columns,
        'item_fields_size': item_fields_size,
    }
    return feat_config
