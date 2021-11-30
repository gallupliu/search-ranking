import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
import tensorflow.feature_column as fc
from models.keras.layers.modules import DNN, MultiHeadAttention
from models.keras.layers.transformer import Encoder

class Que2Search(Model):

    def __init__(self, config,
                 #   num_sampled=1,
                 #   user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32), dnn_activation='relu',
                 # l2_reg_embedding=1e-6, dnn_dropout=0,
                 **kwargs):
        super(Que2Search, self).__init__(**kwargs)
        self.num_sampled = config['num_sampled']
        user_dnn_hidden_units = (64, 32)
        item_dnn_hidden_units = (64, 32)
        dnn_activation = 'relu'
        dnn_dropout = 0
        self.embed_size = config['embed_size']

        self.user_cols = config['user_cols']
        self.item_cols = config['item_cols']
        self.text_cols = config['text_cols']
        self.categorical_cols = config['categorical_cols']
        self.numeric_cols = config['numeric_cols']
        self.bucket_cols = config['bucket_cols']
        self.crossed_cols = config['crossed_cols']
        self.l2_reg_embedding = config['l2_reg_embedding']

        # [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
        text_embed = Embedding(input_dim=config['vocab_size'] + 1,
                               # input_length=feat['feat_len'],
                               output_dim=config['embed_size'],
                               embeddings_initializer='random_uniform',
                               embeddings_regularizer=l2(
                                   self.l2_reg_embedding)
                               )
        user_encoder=  Encoder(num_layers=2, d_model=128, num_heads=8,
                         dff=2048, input_vocab_size=21128,
                         maximum_position_encoding=10000)

        item_encoder = Encoder(num_layers=6, d_model=128, num_heads=8,
                               dff=2048, input_vocab_size=21128,
                               maximum_position_encoding=10000)
        self.avg_embedding = keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))
        self.pooling_embedding = GlobalAveragePooling1D()

        self.user_embed_layers = {}
        for feat in self.user_cols:
            print('user feat:{0}'.format(feat))
            if feat['name'] not in self.text_cols:
                self.user_embed_layers['embed_' + str(feat['name'])] = Embedding(input_dim=feat['num'],
                                                                                 # input_length=feat['feat_len'],
                                                                                 output_dim=feat['embed_dim'],
                                                                                 embeddings_initializer='random_uniform',
                                                                                 embeddings_regularizer=l2(
                                                                                     self.l2_reg_embedding))
            else:
                self.user_embed_layers['embed_' + str(feat['name'])] = text_embed

            self.item_embed_layers = {}
            for feat in self.item_cols:
                print('item feat:{0}'.format(feat))
                if feat['name'] not in self.text_cols:
                    input_dim = feat['num']
                    if feat['name'] in self.categorical_cols:
                        input_dim = feat['num'] + 1
                    self.item_embed_layers['embed_' + str(feat['name'])] = Embedding(input_dim=input_dim,
                                                                                     # input_length=feat['feat_len'],
                                                                                     output_dim=feat['embed_dim'],
                                                                                     embeddings_initializer='random_uniform',
                                                                                     embeddings_regularizer=l2(
                                                                                         self.l2_reg_embedding))
                else:
                    self.item_embed_layers['embed_' + str(feat['name'])] = text_embed

        self.user_dnn = DNN(user_dnn_hidden_units, dnn_activation, dnn_dropout)
        self.item_dnn = DNN(item_dnn_hidden_units, dnn_activation, dnn_dropout)

    # def cosine_similarity(self, tensor1, tensor2):
    #     """计算cosine similarity"""
    #     # 把张量拉成矢量，这是我自己的应用需求
    #     # tensor1 = tf.reshape(tensor1, shape=(1, -1))
    #     # tensor2 = tf.reshape(tensor2, shape=(1, -1))
    #     # 求模长
    #     tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1)))
    #     tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2)))
    #     # 内积
    #     tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2))
    #     # cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
    #     cosin = tf.realdiv(tensor1_tensor2, tensor1_norm * tensor2_norm)
    #
    #     return cosin

    def cosine_similarity(self, query_emb, target_emb):
        query_emb = tf.nn.l2_normalize(query_emb, 0)
        target_emb = tf.nn.l2_normalize(target_emb, 0),
        consine = tf.keras.losses.cosine_similarity(query_emb, target_emb)
        print("Cosine Similarity:", consine)

        # Normalized Euclidean Distance
        # s = tf.norm(tf.nn.l2_normalize(x1, 0) - tf.nn.l2_normalize(y1, 0), ord='euclidean')
        # print("Normalized Euclidean Distance:", s)

        return consine

    def call(self, inputs, training=False, mask=None):

        user_inputs, item_inputs = inputs

        print('user_inputs:{0}'.format(user_inputs))

        user_embedding = []
        for k, v in user_inputs.items():
            print('col:{0},input:{1},embed:{2}'.format(k, v, 'embed_{}'.format(k)))
            print('embed_{}'.format(k))
            user_col_embed = self.user_embed_layers['embed_{}'.format(k)](v)
            if k in self.text_cols:
                user_avg_embed = self.pooling_embedding(user_col_embed)
                user_col_embed = tf.reshape(user_avg_embed, [-1, 1, self.embed_size])
            user_embedding.append(user_col_embed)
            # print('embed_{0},{1}'.format(k,col_embed))

        # user_sparse_embed = tf.concat([self.user_embed_layers['embed_{}'.format(k)](v)
        #                                for k, v in user_inputs.items()], axis=-1)
        user_sparse_embed = tf.concat(user_embedding, axis=-1)

        user_dnn_input = user_sparse_embed
        self.user_dnn_out = self.user_dnn(user_dnn_input)

        print('item_inputs:{0}'.format(item_inputs))
        item_embedding = []
        item_feature_columns = []
        item_feature_inputs = {}
        for k, v in item_inputs.items():
            print('col:{0},input:{1},embed:{2}'.format(k, v, 'embed_{}'.format(k)))
            print('embed_{}'.format(k))
            feat = {}
            for i in range(len(self.item_cols)):
                if self.item_cols[i]['name'] == k:
                    feat = self.item_cols[i]

            if k in self.text_cols:
                item_col_embed = self.item_embed_layers['embed_{}'.format(k)](v)
                item_col_embed = self.pooling_embedding(item_col_embed)
                item_col_embed = tf.reshape(item_col_embed, [-1, 1, self.embed_size])
            elif k in self.categorical_cols:
                item_feature_inputs[k] = item_inputs[k]
                category = fc.categorical_column_with_vocabulary_list(
                        k, feat['vocab_list'])
                category_column = fc.embedding_column(category, feat['embed_dim'])
                item_feature_columns.append(category_column)

                # print('category_column:{0}'.format(category_column))
                # category_feature_layer = tf.keras.layers.DenseFeatures(category_column)
                # category_feature_outputs = category_feature_layer(item_inputs)
                # print('category_feature_outputs{0}'.format(category_feature_outputs))

            elif k in self.numeric_cols:
                item_feature_inputs[k] = item_inputs[k]
                feat_col = fc.numeric_column(feat['name'])
                item_feature_columns.append(feat_col)

                # print('feat_col:{0}'.format(feat_col))
                # feat_col_layer = tf.keras.layers.DenseFeatures(feat_col)
                # feat_col_outputs = feat_col_layer(item_inputs)
                # print('feat_col_outputs {0}'.format(feat_col_outputs ))

            if k in self.bucket_cols:
                item_feature_inputs[k] = item_inputs[k]
                feat_buckets = fc.bucketized_column(feat_col, boundaries=feat['bins'])
                item_feature_columns.append(feat_buckets)
                # print('bucket feat_col:{0}'.format(feat_buckets))
                # feat_buckets_layer = tf.keras.layers.DenseFeatures(feat_buckets)
                # feat_buckets_outputs = feat_buckets_layer(item_inputs)
                # print('feat_buckets_outputs {0}'.format(feat_buckets_outputs))

            item_embedding.append(item_col_embed)
        print('item_feature_columns:{0}'.format(item_feature_columns))
        feature_layer = tf.keras.layers.DenseFeatures(item_feature_columns)
        feature_layer_outputs = tf.expand_dims(feature_layer(item_feature_inputs), axis=1)
        print('item_embed:{0}'.format(item_embedding))
        print('feature_layer_outputs:{0}'.format(feature_layer_outputs))
        item_embedding.append(feature_layer_outputs)

        print('embed_user:{0},item:{1}'.format(user_embedding, item_embedding))
        item_sparse_embed = tf.concat(item_embedding, axis=-1)
        item_dnn_input = item_sparse_embed
        self.item_dnn_out = self.item_dnn(item_dnn_input)

        print('item_dnn_out:{0}'.format(self.item_dnn_out))
        print('user_dnn_out:{0}'.format(self.user_dnn_out))
        cosine_score = tf.sigmoid(self.cosine_similarity(self.item_dnn_out, self.user_dnn_out),name='cosine_score')
        output = tf.reshape(cosine_score, (-1, ), name='score')
        print('cosine_score:{0}'.format(cosine_score))
        print('output:{0}'.format(output))

        return output

    def summary(self, **kwargs):
        user_inputs = {}
        item_inputs = {}

        for feat in self.user_cols + self.item_cols:

            print('feat input:{0}'.format(feat))
            if feat['name'] in self.text_cols:
                #feature_column 共享embedding存在问题，暂时用传统方法
                feat_input = Input(shape=(feat['num'],), name=feat['name'], dtype='int32')
            elif feat['name'] in self.categorical_cols:
                feat_input = Input(shape=(1,), name=feat['name'], dtype='string')

            else:
                feat_input = Input(shape=(1,), name=feat['name'])

            if feat in self.user_cols:
                user_inputs[feat['name']] = feat_input
            else:
                item_inputs[feat['name']] = feat_input


        model = Model(inputs=[user_inputs, item_inputs],
                      outputs=self.call([user_inputs, item_inputs]))

        model.__setattr__("user_input", user_inputs)
        model.__setattr__("item_input", item_inputs)
        model.__setattr__("user_embeding", self.user_dnn_out)
        model.__setattr__("item_embeding", self.item_dnn_out)
        return model
