import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
import tensorflow.feature_column as fc
from models.keras.layers.modules import DNN, MultiHeadAttention, NegativeCosineLayer, Similarity


#
#
# class BST_DSSM(tf.keras.Model):
#     """define BST+DSSM model stucture
#     用subclass的方法来定义模型
#     """
#     def __init__(self,
#                  config,
#                  # item_embedding=None, user_embedding=None,
#                  # emb_dim=emb_dim,
#                  # vocab_size=vocab_size,
#                  # item_max_len=item_max_len, user_max_len=user_max_len,
#                  # epoch=10, batch_size=batch_size, n_layers=n_layer,
#                  # learning_rate=lr, optimizer_type="adam",
#                  random_seed=2019,
#                  has_residual=True):
#         """
#         initial model related parms and tensors
#         """
#         super(BST_DSSM, self).__init__()
#         self.emb_dim = config[]
#
#         self.l2_reg = l2_reg
#
#         self.learning_rate = learning_rate
#         self.optimizer_type = optimizer_type
#
#         self.blocks = config["num_layers"]
#
#         self.random_seed = random_seed
#
#         self.vocab_size = config["vocab_size"]
#         self.item_text_len =
#         self.item_max_len = item_max_len
#         self.user_max_len = user_max_len
#         self.has_residual = has_residual
#
#         self.mha_user = MultiHeadAttention(scope_name="user", embed_dim=self.emb_dim)
#         self.mha_item = MultiHeadAttention(scope_name="item", embed_dim=self.emb_dim)
#
#         # optimizer
#         if self.optimizer_type == "adam":
#             self.optimizer = tf.keras.optimizers.Adam(
#                 learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#         elif self.optimizer_type == "adagrad":
#             self.optimizer = tf.keras.optimizers.Adagrad(
#                 learning_rate=self.learning_rate,
#                 initial_accumulator_value=1e-8)
#         elif self.optimizer_type == "gd":
#             self.optimizer = tf.keras.optimizers.SGD(
#                 learning_rate=self.learning_rate)
#         elif self.optimizer_type == "momentum":
#             self.optimizer = tf.keras.optimizers.SGD(
#                 learning_rate=self.learning_rate, momentum=0.95)
#
#         self.user_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.emb_dim)
#         self.item_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.emb_dim)
#
#     @tf.function
#     def call(self, inputs, training=True):
#         # multiple inputs
#         item_inputs = inputs[0]
#         user_inputs = inputs[1]
#
#         # if feature_type == "cnn":
#         #     utterance_cnn = tf.keras.layers.Conv1D(filters=M_dim,
#         #                                             kernel_size=2,
#         #                                             padding='same',
#         #                                             activation=tf.tanh,
#         #                                             name='utterance_conv')
#
#         # 用户和物品各用一个lut,类似DSSM中的双塔
#         # 维度变成[batch_size, max_length, emb_dim]
#         item_sequence_embeddings = self.item_embedding(item_inputs)
#         user_sequence_embeddings = self.user_embedding(user_inputs)
#
#         # mask
#         item_mask = tf.where(
#             item_inputs == 0,
#             x=tf.ones_like(item_inputs, dtype=tf.float32),
#             y=tf.zeros_like(item_inputs, dtype=tf.float32))
#         item_mask = tf.expand_dims(item_mask, axis=-1)
#         user_mask = tf.where(
#             user_inputs == 0,
#             x=tf.ones_like(user_inputs, dtype=tf.float32),
#             y=tf.zeros_like(user_inputs, dtype=tf.float32))
#         user_mask = tf.expand_dims(user_mask, axis=-1)
#
#         # 维度变成[batch_size, max_length, 16]
#         for i in range(self.blocks):
#             item_sequence_embeddings = self.mha_item(item_sequence_embeddings, src_mask=item_mask)
#             user_sequence_embeddings = self.mha_user(user_sequence_embeddings, src_mask=user_mask)
#
#         # 最大池化层, 维度变成[batch_size, 1, 16]
#         item_outputs_max = tf.nn.max_pool(
#             item_sequence_embeddings,
#             [1, self.item_max_len, 1],
#             [1 for _ in range(len(item_sequence_embeddings.shape))],
#             padding="VALID")
#         user_outputs_max = tf.nn.max_pool(
#             user_sequence_embeddings,
#             [1, self.user_max_len, 1],
#             [1 for _ in range(len(user_sequence_embeddings.shape))],
#             padding="VALID")
#
#         # 向量归一化用于计算cosine相似度
#         item_normalized = tf.nn.l2_normalize(
#             item_outputs_max, axis=2)
#         user_normalized = tf.nn.l2_normalize(
#             user_outputs_max, axis=2)
#
#         # cosine相似度
#         outputs = tf.matmul(
#             item_normalized,
#             user_normalized,
#             transpose_b=True)
#         return tf.reshape(outputs, [-1, 1])
#
#     def loss_fn(self, target, output):
#         cross_entropy = tf.keras.backend.binary_crossentropy(
#             target, output, from_logits=False
#         )
#         if self.l2_reg > 0:
#             _regularizer = tf.keras.regularizers.l2(self.l2_reg)
#             cross_entropy += _regularizer(self.user_embedding)
#             cross_entropy += _regularizer(self.item_embedding)
#         return cross_entropy
#
#     def focal_loss(self, target, output):
#         target = tf.reshape(target, [-1, 1])
#         y = tf.multiply(output, target) + tf.multiply(1 - output, 1 - target)
#         loss = tf.pow(1. - y, self.gamma) * tf.math.log(y + self.epsilon)
#         return - tf.reduce_mean(loss)

# def DSSM(
#         config):
#     num_sampled = config['num_sampled']
#     user_dnn_hidden_units = config['user_dnn_hidden_units']
#     item_dnn_hidden_units = config['item_dnn_hidden_units']
#     dnn_activation = config['out_dnn_activation']
#     dnn_dropout = config['dnn_dropout']
#     embed_size = config['embed_size']
#
#     user_cols = config['user_cols']
#     item_cols = config['item_cols']
#     text_cols = config['text_cols']
#     categorical_cols = config['categorical_cols']
#     numeric_cols = config['numeric_cols']
#     bucket_cols = config['bucket_cols']
#     crossed_cols = config['crossed_cols']
#     l2_reg_embedding = config['l2_reg_embedding']
#     user_inputs = {}
#     item_inputs = {}
#
#     #构建输入
#     for feat in user_cols:
#
#         print('feat input:{0}'.format(feat))
#         if feat['name'] in text_cols:
#             # feature_column 共享embedding存在问题，暂时用传统方法
#             feat_input = Input(shape=(feat['num'],), name=feat['name'], dtype='int32')
#         elif feat['name'] in categorical_cols:
#             feat_input = Input(shape=(1,), name=feat['name'], dtype='string')
#
#         else:
#             feat_input = Input(shape=(1,), name=feat['name'])
#
#         user_inputs[feat['name']] = feat_input
#
#     for feat in item_cols:
#
#         print('feat input:{0}'.format(feat))
#         if feat['name'] in text_cols:
#             # feature_column 共享embedding存在问题，暂时用传统方法
#             feat_input = Input(shape=(feat['num'],), name=feat['name'], dtype='int32')
#         elif feat['name'] in categorical_cols:
#             feat_input = Input(shape=(1,), name=feat['name'], dtype='string')
#
#         else:
#             feat_input = Input(shape=(1,), name=feat['name'])
#
#         if feat in user_cols:
#             user_inputs[feat['name']] = feat_input
#         else:
#             item_inputs[feat['name']] = feat_input
#
#
#
#     model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)
#     model.__setattr__("user_input", user_inputs_list)
#     model.__setattr__("item_input", item_inputs_list)
#     model.__setattr__("user_embedding", user_dnn_out)
#     model.__setattr__("item_embedding", item_dnn_out)
#
#     return model

class DSSMModel(tf.keras.Model):
    def __init__(self, config):
        super(DSSMModel, self).__init__()

        self.num_sampled = config['num_sampled']
        self.user_dnn_hidden_units = config['user_dnn_hidden_units']
        self.item_dnn_hidden_units = config['item_dnn_hidden_units']
        dnn_activation = config['out_dnn_activation']
        dnn_dropout = config['dnn_dropout']
        self.embed_size = config['embed_size']

        self.user_cols = config['user_cols']
        self.item_cols = config['item_cols']
        self.text_cols = config['text_cols']
        self.categorical_cols = config['categorical_cols']
        self.numeric_cols = config['numeric_cols']
        self.bucket_cols = config['bucket_cols']
        self.crossed_cols = config['crossed_cols']
        self.l2_reg_embedding = config['l2_reg_embedding']
        self.config = config

        user_inputs = {}
        item_inputs = {}

        for feat in self.user_cols:

            print('feat input:{0}'.format(feat))
            if feat['name'] in self.text_cols:
                # feature_column 共享embedding存在问题，暂时用传统方法
                feat_input = Input(shape=(feat['num'],), name=feat['name'], dtype='int64')
            elif feat['name'] in self.categorical_cols:
                feat_input = Input(shape=(1,), name=feat['name'], dtype='string')

            else:
                feat_input = Input(shape=(1,), name=feat['name'])

            user_inputs[feat['name']] = feat_input

        for feat in self.item_cols:

            print('feat input:{0}'.format(feat))
            if feat['name'] in self.text_cols:
                # feature_column 共享embedding存在问题，暂时用传统方法
                feat_input = Input(shape=(feat['num'],), name=feat['name'], dtype='int64')
            elif feat['name'] in self.categorical_cols:
                feat_input = Input(shape=(1,), name=feat['name'], dtype='string')

            else:
                feat_input = Input(shape=(1,), name=feat['name'])

            if feat in self.user_cols:
                user_inputs[feat['name']] = feat_input
            else:
                item_inputs[feat['name']] = feat_input

        # [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
        text_embed = Embedding(input_dim=config['vocab_size'] + 1,
                               # input_length=feat['feat_len'],
                               output_dim=config['embed_size'],
                               embeddings_initializer='random_uniform',
                               embeddings_regularizer=l2(
                                   self.l2_reg_embedding)
                               )
        self.avg_embedding = keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))
        self.pooling_embedding = GlobalAveragePooling1D()
        # self.negativecosine_layer = NegativeCosineLayer(self.config['neg'],self.config['batch_size'])

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

        self.user_dnn = DNN(self.user_dnn_hidden_units, dnn_activation, dnn_dropout)
        self.item_dnn = DNN(self.item_dnn_hidden_units, dnn_activation, dnn_dropout)

        self.model = keras.Model(inputs=[user_inputs, item_inputs], outputs=self.call([user_inputs, item_inputs]))

    def call(self, inputs, training=False):
        print('inputs:{0}'.format(inputs))

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
        self.user_dnn_out = tf.reshape(self.user_dnn_out, [-1, self.item_dnn_hidden_units[-1]])
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
        # print('item_feature_columns:{0}'.format(item_feature_columns))
        if len(item_feature_columns) > 0:
            feature_layer = tf.keras.layers.DenseFeatures(item_feature_columns)
            feature_layer_outputs = tf.expand_dims(feature_layer(item_feature_inputs), axis=1)
            # print('item_embed:{0}'.format(item_embedding))
            # print('feature_layer_outputs:{0}'.format(feature_layer_outputs))
            item_embedding.append(feature_layer_outputs)

        # print('embed_user:{0},item:{1}'.format(user_embedding, item_embedding))
        item_sparse_embed = tf.concat(item_embedding, axis=-1)
        item_dnn_input = item_sparse_embed
        self.item_dnn_out = self.item_dnn(item_dnn_input)
        self.item_dnn_out = tf.reshape(self.item_dnn_out, [-1, self.item_dnn_hidden_units[-1]])

        score = Similarity(type_sim='cos', gamma=20, name='dssm_out')([self.user_dnn_out, self.item_dnn_out])

        output = score

        return output


def model_fn(config):
    num_sampled = config['num_sampled']
    user_dnn_hidden_units = config['user_dnn_hidden_units']
    item_dnn_hidden_units = config['item_dnn_hidden_units']
    dnn_activation = config['out_dnn_activation']
    dnn_dropout = config['dnn_dropout']
    embed_size = config['embed_size']

    user_cols = config['user_cols']
    item_cols = config['item_cols']
    text_cols = config['text_cols']
    categorical_cols = config['categorical_cols']
    numeric_cols = config['numeric_cols']
    bucket_cols = config['bucket_cols']
    crossed_cols = config['crossed_cols']
    l2_reg_embedding = config['l2_reg_embedding']
    config = config

    user_inputs = {}
    item_inputs = {}

    for feat in item_cols:

        print('feat input:{0}'.format(feat))
        if feat['name'] in text_cols:
            # feature_column 共享embedding存在问题，暂时用传统方法
            feat_input = Input(shape=(feat['num'],), name=feat['name'], dtype='int64')
        elif feat['name'] in categorical_cols:
            feat_input = Input(shape=(1,), name=feat['name'], dtype='string')

        else:
            feat_input = Input(shape=(1,), name=feat['name'])

        # if feat in user_cols:
        #     user_inputs[feat['name']] = feat_input
        # else:
        item_inputs[feat['name']] = feat_input

    for feat in user_cols:

        print('feat input:{0}'.format(feat))
        if feat['name'] in text_cols:
            # feature_column 共享embedding存在问题，暂时用传统方法
            feat_input = Input(shape=(feat['num'],), name=feat['name'], dtype='int64')
        elif feat['name'] in categorical_cols:
            feat_input = Input(shape=(1,), name=feat['name'], dtype='string')

        else:
            feat_input = Input(shape=(1,), name=feat['name'])

        user_inputs[feat['name']] = feat_input

    # [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
    text_embed = Embedding(input_dim=config['vocab_size'] + 1,
                           # input_length=feat['feat_len'],
                           output_dim=config['embed_size'],
                           embeddings_initializer='random_uniform',
                           embeddings_regularizer=l2(
                               l2_reg_embedding)
                           )
    avg_embedding = keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))
    pooling_embedding = GlobalAveragePooling1D()
    # negativecosine_layer = NegativeCosineLayer(config['neg'],config['batch_size'])

    user_embed_layers = {}
    for feat in user_cols:
        print('user feat:{0}'.format(feat))
        if feat['name'] not in text_cols:
            user_embed_layers['embed_' + str(feat['name'])] = Embedding(input_dim=feat['num'],
                                                                        # input_length=feat['feat_len'],
                                                                        output_dim=feat['embed_dim'],
                                                                        embeddings_initializer='random_uniform',
                                                                        embeddings_regularizer=l2(
                                                                            l2_reg_embedding))
        else:
            user_embed_layers['embed_' + str(feat['name'])] = text_embed

    item_embed_layers = {}
    for feat in item_cols:
        print('item feat:{0}'.format(feat))
        if feat['name'] not in text_cols:
            input_dim = feat['num']
            if feat['name'] in categorical_cols:
                input_dim = feat['num'] + 1
            item_embed_layers['embed_' + str(feat['name'])] = Embedding(input_dim=input_dim,
                                                                        # input_length=feat['feat_len'],
                                                                        output_dim=feat['embed_dim'],
                                                                        embeddings_initializer='random_uniform',
                                                                        embeddings_regularizer=l2(
                                                                            l2_reg_embedding))
        else:
            item_embed_layers['embed_' + str(feat['name'])] = text_embed

    user_dnn = DNN(user_dnn_hidden_units, dnn_activation, dnn_dropout)
    item_dnn = DNN(item_dnn_hidden_units, dnn_activation, dnn_dropout)

    # user_inputs, item_inputs = inputs

    print('user_inputs:{0}'.format(user_inputs))

    user_embedding = []
    for k, v in user_inputs.items():
        print('col:{0},input:{1},embed:{2}'.format(k, v, 'embed_{}'.format(k)))
        print('embed_{}'.format(k))
        user_col_embed = user_embed_layers['embed_{}'.format(k)](v)
        if k in text_cols:
            user_avg_embed = pooling_embedding(user_col_embed)
            user_col_embed = tf.reshape(user_avg_embed, [-1, 1, embed_size])
        user_embedding.append(user_col_embed)
        # print('embed_{0},{1}'.format(k,col_embed))

    # user_sparse_embed = tf.concat([user_embed_layers['embed_{}'.format(k)](v)
    #                                for k, v in user_inputs.items()], axis=-1)
    user_sparse_embed = tf.concat(user_embedding, axis=-1)

    user_dnn_input = user_sparse_embed
    user_dnn_out = user_dnn(user_dnn_input)
    user_dnn_out = tf.reshape(user_dnn_out, [-1, item_dnn_hidden_units[-1]])

    print('item_inputs:{0}'.format(item_inputs))
    item_embedding = []
    item_feature_columns = []
    item_feature_inputs = {}
    for k, v in item_inputs.items():
        print('col:{0},input:{1},embed:{2}'.format(k, v, 'embed_{}'.format(k)))
        print('embed_{}'.format(k))
        feat = {}
        for i in range(len(item_cols)):
            if item_cols[i]['name'] == k:
                feat = item_cols[i]

        if k in text_cols:
            item_col_embed = item_embed_layers['embed_{}'.format(k)](v)
            item_col_embed = pooling_embedding(item_col_embed)
            item_col_embed = tf.reshape(item_col_embed, [-1, 1, embed_size])
        elif k in categorical_cols:
            item_feature_inputs[k] = item_inputs[k]
            category = fc.categorical_column_with_vocabulary_list(
                k, feat['vocab_list'])
            category_column = fc.embedding_column(category, feat['embed_dim'])
            item_feature_columns.append(category_column)

            # print('category_column:{0}'.format(category_column))
            # category_feature_layer = tf.keras.layers.DenseFeatures(category_column)
            # category_feature_outputs = category_feature_layer(item_inputs)
            # print('category_feature_outputs{0}'.format(category_feature_outputs))

        elif k in numeric_cols:
            item_feature_inputs[k] = item_inputs[k]
            feat_col = fc.numeric_column(feat['name'])
            item_feature_columns.append(feat_col)

            # print('feat_col:{0}'.format(feat_col))
            # feat_col_layer = tf.keras.layers.DenseFeatures(feat_col)
            # feat_col_outputs = feat_col_layer(item_inputs)
            # print('feat_col_outputs {0}'.format(feat_col_outputs ))

        if k in bucket_cols:
            item_feature_inputs[k] = item_inputs[k]
            feat_buckets = fc.bucketized_column(feat_col, boundaries=feat['bins'])
            item_feature_columns.append(feat_buckets)
            # print('bucket feat_col:{0}'.format(feat_buckets))
            # feat_buckets_layer = tf.keras.layers.DenseFeatures(feat_buckets)
            # feat_buckets_outputs = feat_buckets_layer(item_inputs)
            # print('feat_buckets_outputs {0}'.format(feat_buckets_outputs))

        item_embedding.append(item_col_embed)
    if len(item_feature_columns) > 0:
        # print('item_feature_columns:{0}'.format(item_feature_columns))
        feature_layer = tf.keras.layers.DenseFeatures(item_feature_columns)
        feature_layer_outputs = tf.expand_dims(feature_layer(item_feature_inputs), axis=1)
        # print('item_embed:{0}'.format(item_embedding))
        # print('feature_layer_outputs:{0}'.format(feature_layer_outputs))
        item_embedding.append(feature_layer_outputs)

    # print('embed_user:{0},item:{1}'.format(user_embedding, item_embedding))
    item_sparse_embed = tf.concat(item_embedding, axis=-1)
    item_dnn_input = item_sparse_embed
    item_dnn_out = item_dnn(item_dnn_input)
    item_dnn_out = tf.reshape(item_dnn_out, [-1, item_dnn_hidden_units[-1]])

    score = Similarity(type_sim='cos', gamma=20, name='dssm_out')([user_dnn_out, item_dnn_out])

    output = score
    print('user inputs:{0} item inputs:{1}'.format(user_inputs, item_inputs))
    # [user_inputs, item_inputs]
    inputs = []
    for key, value in user_inputs.items():
        print(value)
        inputs.append(value)
    for key, value in item_inputs.items():
        print(value)
        inputs.append(value)
    print(type(inputs), inputs)
    model = keras.Model(inputs=inputs, outputs=output)
    return model


class DSSM(Model):

    def __init__(self, config,
                 #   num_sampled=1,
                 #   user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32), dnn_activation='relu',
                 # l2_reg_embedding=1e-6, dnn_dropout=0,
                 **kwargs):
        super(DSSM, self).__init__(**kwargs)
        self.num_sampled = config['num_sampled']
        # user_dnn_hidden_units = (64, 32)
        # item_dnn_hidden_units = (64, 32)
        # dnn_activation = 'relu'
        # dnn_dropout = 0
        self.user_dnn_hidden_units = config['user_dnn_hidden_units']
        self.item_dnn_hidden_units = config['item_dnn_hidden_units']
        dnn_activation = config['out_dnn_activation']
        dnn_dropout = config['dnn_dropout']
        self.embed_size = config['embed_size']

        self.user_cols = config['user_cols']
        self.item_cols = config['item_cols']
        self.text_cols = config['text_cols']
        self.categorical_cols = config['categorical_cols']
        self.numeric_cols = config['numeric_cols']
        self.bucket_cols = config['bucket_cols']
        self.crossed_cols = config['crossed_cols']
        self.l2_reg_embedding = config['l2_reg_embedding']
        self.config = config

        # [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
        text_embed = Embedding(input_dim=config['vocab_size'] + 1,
                               # input_length=feat['feat_len'],
                               output_dim=config['embed_size'],
                               embeddings_initializer='random_uniform',
                               embeddings_regularizer=l2(
                                   self.l2_reg_embedding)
                               )
        self.avg_embedding = keras.layers.Lambda(lambda x: tf.reduce_mean(x, 1))
        self.pooling_embedding = GlobalAveragePooling1D()
        # self.negativecosine_layer = NegativeCosineLayer(self.config['neg'],self.config['batch_size'])

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

        self.user_dnn = DNN(self.user_dnn_hidden_units, dnn_activation, dnn_dropout)
        self.item_dnn = DNN(self.item_dnn_hidden_units, dnn_activation, dnn_dropout)

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
        self.user_dnn_out = tf.reshape(self.user_dnn_out, [-1, self.item_dnn_hidden_units[-1]])
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
        # print('item_feature_columns:{0}'.format(item_feature_columns))
        feature_layer = tf.keras.layers.DenseFeatures(item_feature_columns)
        feature_layer_outputs = tf.expand_dims(feature_layer(item_feature_inputs), axis=1)
        # print('item_embed:{0}'.format(item_embedding))
        # print('feature_layer_outputs:{0}'.format(feature_layer_outputs))
        item_embedding.append(feature_layer_outputs)

        # print('embed_user:{0},item:{1}'.format(user_embedding, item_embedding))
        item_sparse_embed = tf.concat(item_embedding, axis=-1)
        item_dnn_input = item_sparse_embed
        self.item_dnn_out = self.item_dnn(item_dnn_input)
        self.item_dnn_out = tf.reshape(self.item_dnn_out, [-1, self.item_dnn_hidden_units[-1]])

        score = Similarity(type_sim='cos', gamma=20, name='dssm_out')([self.user_dnn_out, self.item_dnn_out])

        output = score

        # # # 随机采样负样本
        # with tf.name_scope("rotate"):
        #     tmp = tf.tile(self.item_dnn_out, [1, 1])
        #     item_encoder_fd = self.item_dnn_out
        #     for i in range( self.config['neg']):
        #         rand = tf.cast(((random.random() + i) * tf.cast(self.config['batch_size'], tf.float32) /self.config['neg']), tf.int32)
        #         item_encoder_fd = tf.concat([item_encoder_fd,
        #                                      tf.slice(tmp, [rand, 0], [self.config['batch_size']- rand, -1]),
        #                                      tf.slice(tmp, [0, 0], [rand, -1])], axis=0)
        # # Cosine similarity
        # with tf.name_scope("cosine_similarity"):
        #     user_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.user_dnn_out), axis=1, keepdims=True)),
        #                         [self.config['neg'] + 1, 1])
        #     item_norm = tf.sqrt(tf.reduce_sum(tf.square(item_encoder_fd), axis=1, keepdims=True))
        #     # prod [(NEG + 1) * batch_size, 1] tf.tile对数据进行复制
        #     prod = tf.reduce_sum(tf.multiply(tf.tile(self.user_dnn_out, [self.config['neg'] + 1, 1]), item_encoder_fd), axis=1,
        #                          keepdims=True)
        #     norm_prod = tf.multiply(user_norm, item_norm)
        #     cos_sim_raw = tf.truediv(prod, norm_prod)
        #     cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [self.config['neg'] + 1, -1])) * 20
        #
        # # 最大化正样本概率
        # with tf.name_scope("loss"):
        #     prob = tf.nn.softmax(cos_sim)
        #     hit_prob = tf.slice(prob, [0, 0], [-1, 1])
        #     self.loss = -tf.reduce_mean(tf.math.log(hit_prob))
        #     correct_prediction = tf.cast(tf.equal(tf.argmax(prob, 1), 0), tf.float32)
        #     accuracy = tf.reduce_mean(correct_prediction)
        #     output = hit_prob
        # cosine_score = tf.sigmoid(self.negativecosine_layer([self.item_dnn_out, self.user_dnn_out]))

        # cosine_score = tf.sigmoid(self.cosine_similarity(self.item_dnn_out, self.user_dnn_out), name='cosine_score')
        # output = tf.reshape(cosine_score, (-1,), name='score')
        # print('cosine_score:{0}'.format(cosine_score))
        # print('output:{0}'.format(output))

        return output

    def build_model(self, **kwargs):
        user_inputs = {}
        item_inputs = {}

        # for feat in self.user_cols + self.item_cols:
        #
        #     print('feat input:{0}'.format(feat))
        #     if feat['name'] in self.text_cols:
        #         #feature_column 共享embedding存在问题，暂时用传统方法
        #         feat_input = Input(shape=(feat['num'],), name=feat['name'], dtype='int32')
        #     elif feat['name'] in self.categorical_cols:
        #         feat_input = Input(shape=(1,), name=feat['name'], dtype='string')
        #
        #     else:
        #         feat_input = Input(shape=(1,), name=feat['name'])
        #
        #     if feat in self.user_cols:
        #         user_inputs[feat['name']] = feat_input
        #     else:
        #         item_inputs[feat['name']] = feat_input

        for feat in self.user_cols:

            print('feat input:{0}'.format(feat))
            if feat['name'] in self.text_cols:
                # feature_column 共享embedding存在问题，暂时用传统方法
                feat_input = Input(shape=(feat['num'],), name=feat['name'], dtype='int32')
            elif feat['name'] in self.categorical_cols:
                feat_input = Input(shape=(1,), name=feat['name'], dtype='string')

            else:
                feat_input = Input(shape=(1,), name=feat['name'])

            user_inputs[feat['name']] = feat_input

        for feat in self.item_cols:

            print('feat input:{0}'.format(feat))
            if feat['name'] in self.text_cols:
                # feature_column 共享embedding存在问题，暂时用传统方法
                feat_input = Input(shape=(feat['num'],), name=feat['name'], dtype='int32')
            elif feat['name'] in self.categorical_cols:
                feat_input = Input(shape=(1,), name=feat['name'], dtype='string')

            else:
                feat_input = Input(shape=(1,), name=feat['name'])

            if feat in self.user_cols:
                user_inputs[feat['name']] = feat_input
            else:
                item_inputs[feat['name']] = feat_input

        inputs = []
        for key, value in user_inputs.items():
            print(value)
            inputs.append(value)
        for key, value in item_inputs.items():
            print(value)
            inputs.append(value)

        model = Model(inputs=inputs,
                      outputs=self.call([user_inputs, item_inputs]))

        model.__setattr__("user_input", user_inputs)
        model.__setattr__("item_input", item_inputs)
        model.__setattr__("user_embeding", self.user_dnn_out)
        model.__setattr__("item_embeding", self.item_dnn_out)
        return model
