import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from models.keras.layers.modules import DNN, MultiHeadAttention


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

class DSSM(Model):

    def __init__(self, config,
                 #   num_sampled=1,
                 #   user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32), dnn_activation='relu',
                 # l2_reg_embedding=1e-6, dnn_dropout=0,
                 **kwargs):
        super(DSSM, self).__init__(**kwargs)
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
        self.l2_reg_embedding = config['l2_reg_embedding']

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

    def cosine_similarity(self, tensor1, tensor2):
        """计算cosine similarity"""
        # 把张量拉成矢量，这是我自己的应用需求
        tensor1 = tf.reshape(tensor1, shape=(1, -1))
        tensor2 = tf.reshape(tensor2, shape=(1, -1))
        # 求模长
        tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1)))
        tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2)))
        # 内积
        tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2))
        # cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
        cosin = tf.realdiv(tensor1_tensor2, tensor1_norm * tensor2_norm)

        return cosin

    def call(self, inputs, training=None, mask=None):
        print('inputs:{0},item:{1}'.format(inputs, inputs['item']))
        user_inputs = {}
        item_inputs = {}
        for feat in self.user_cols:
            user_inputs[feat['name']] = inputs[feat['name']]
        for feat in self.item_cols:
            if feat['name'] in self.categorical_cols:
                input = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=feat['vocab_list'],
                                                                                num_oov_indices=0)(
                    inputs[feat['name']])
                item_inputs[feat['name']] = input
            elif feat['name'] in self.numeric_cols:
                print(feat['name'])
                # hashed_input = tf.keras.experimental.preprocessing.Hashing(num_bins=hash_bucket_size)(keras_input)
                input = tf.keras.layers.experimental.preprocessing.Discretization(bins=feat['bins'])(
                    inputs[feat['name']])
                item_inputs[feat['name']] = input
            else:
                item_inputs[feat['name']] = inputs[feat['name']]

        # user_inputs, item_inputs = inputs
        user_embedding = []
        for k, v in user_inputs.items():

            col_embed = self.user_embed_layers['embed_{}'.format(k)](v)
            if k in self.text_cols:
                avg_embed = self.pooling_embedding(col_embed)
                col_embed = tf.reshape(avg_embed, [-1, 1, self.embed_size])
            user_embedding.append(col_embed)
            # print('embed_{0},{1}'.format(k,col_embed))

        # user_sparse_embed = tf.concat([self.user_embed_layers['embed_{}'.format(k)](v)
        #                                for k, v in user_inputs.items()], axis=-1)
        user_sparse_embed = tf.concat(user_embedding, axis=-1)

        user_dnn_input = user_sparse_embed
        self.user_dnn_out = self.user_dnn(user_dnn_input)
        item_embedding = []
        for k, v in item_inputs.items():
            print('col:{0},input:{1},embed:{2}'.format(k, v, self.item_embed_layers['embed_{}'.format(k)]))

            col_embed = self.item_embed_layers['embed_{}'.format(k)](v)
            if k in self.text_cols:
                avg_embed = self.pooling_embedding(col_embed)
                col_embed = tf.reshape(avg_embed, [-1, 1, self.embed_size])

            item_embedding.append(col_embed)
            # print('embed_{0},{1}'.format(k,col_embed))

        item_sparse_embed = tf.concat(item_embedding, axis=-1)
        item_dnn_input = item_sparse_embed
        self.item_dnn_out = self.item_dnn(item_dnn_input)

        output = self.cosine_similarity(self.item_dnn_out, self.user_dnn_out)

        return output

    def summary(self, **kwargs):
        print()
        user_inputs = {}
        for feat in self.user_cols:
            print('user input:{0}'.format(feat))
            if feat['name'] in self.text_cols:
                col = Input(shape=(feat['num'],), name=feat['name'], dtype='int32')
            elif feat['name'] in self.categorical_cols:
                col = Input(shape=(1,), name=feat['name'])
                # col = tf.keras.layers.StringLookup(max_tokens=len(feat['vocab_list']) + 1,
                #                                    num_oov_indices=1,
                #                                    mask_token=None, vocabulary=feat['vocab_list'])(
                #     col)
            else:
                col = Input(shape=(1,), name=feat['name'])
            user_inputs[feat['name']] = col

        item_inputs = {}
        for feat in self.item_cols:
            print('item input:{0}'.format(feat))
            if feat['name'] in self.text_cols:
                col = Input(shape=(feat['num'],), name=feat['name'], dtype='int32')
            elif feat['name'] in self.categorical_cols:
                col = Input(shape=(1,), name=feat['name'])
                # col = tf.keras.layers.StringLookup(max_tokens=len(feat['vocab_list']) + 1,
                #                                    num_oov_indices=1,
                #                                    mask_token=None, vocabulary=feat['vocab_list'])(
                #     col)
                print('item input:{0}'.format(col))
            else:
                col = Input(shape=(1,), name=feat['name'])
            item_inputs[feat['name']] = col

        model = Model(inputs=[user_inputs, item_inputs],
                      outputs=self.call([user_inputs, item_inputs]))

        model.__setattr__("user_input", user_inputs)
        model.__setattr__("item_input", item_inputs)
        model.__setattr__("user_embeding", self.user_dnn_out)
        model.__setattr__("item_embeding", self.item_dnn_out)
        return model

# def model_test():
#     user_features = [{'feat': 'user_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
#     item_features = [{'feat': 'item_id', 'feat_num': 100, 'feat_len': 1, 'embed_dim': 8}]
#     model = DSSM(user_features, item_features)
#     model.summary()
#
# model_test()
