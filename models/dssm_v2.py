# import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import *


# 自定义dnese层含BN， dropout
class CustomDense(Layer):
    def __init__(self, units=32, activation='tanh', dropout_rate=0, use_bn=False, seed=1024, tag_name="dnn", **kwargs):
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        self.tag_name = tag_name

        super(CustomDense, self).__init__(**kwargs)

    # build方法一般定义Layer需要被训练的参数。
    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='random_normal',
                                      trainable=True,
                                      name='kernel_' + self.tag_name)
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='random_normal',
                                    trainable=True,
                                    name='bias_' + self.tag_name)

        if self.use_bn:
            self.bn_layers = tf.keras.layers.BatchNormalization()

        self.dropout_layers = tf.keras.layers.Dropout(self.dropout_rate)
        self.activation_layers = tf.keras.layers.Activation(self.activation, name=self.activation + '_' + self.tag_name)

        super(CustomDense, self).build(input_shape)  # 相当于设置self.built = True

    # call方法一般定义正向传播运算逻辑，__call__方法调用了它。
    def call(self, inputs, training=None, **kwargs):
        fc = tf.matmul(inputs, self.weight) + self.bias
        if self.use_bn:
            fc = self.bn_layers(fc)
        out_fc = self.activation_layers(fc)

        return out_fc

    # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法，保存模型不写这部分会报错
    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({'units': self.units, 'activation': self.activation, 'use_bn': self.use_bn,
                       'dropout_rate': self.dropout_rate, 'seed': self.seed, 'name': self.tag_name})
        return config


# cos 相似度计算层
class Similarity(Layer):

    def __init__(self, gamma=1, axis=-1, type_sim='cos', **kwargs):
        self.gamma = gamma
        self.axis = axis
        self.type_sim = type_sim
        super(Similarity, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Similarity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        query, candidate = inputs
        if self.type_sim == "cos":
            query_norm = tf.norm(query, axis=self.axis)
            candidate_norm = tf.norm(candidate, axis=self.axis)
        cosine_score = tf.reduce_sum(tf.multiply(query, candidate), -1)
        cosine_score = tf.divide(cosine_score, query_norm * candidate_norm + 1e-8)
        cosine_score = tf.clip_by_value(cosine_score, -1, 1.0) * self.gamma
        return tf.expand_dims(cosine_score, 1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'gamma': self.gamma, 'axis': self.axis, 'type': self.type_sim}
        base_config = super(Similarity, self).get_config()
        return base_config.uptate(config)


def dssm_model_fn(features, labels, mode, params):
    user_columns = params['user_columns']
    user_fields_size = params['user_fields_size']
    # org_emb_size = params['embedding_dim']
    item_columns = params['item_columns']
    item_fields_size = params['item_fields_size']

    user_dnn_hidden_units = params['user_dnn_hidden_units']
    item_dnn_hidden_units = params['item_dnn_hidden_units']
    user_dnn_dropout = params['user_dnn_dropout']
    item_dnn_dropout = params['item_dnn_dropout']
    out_dnn_activation = params['out_dnn_activation']
    gamma = params['gamma']
    dnn_use_bn = params['dnn_use_bn']
    seed = params['seed']
    metric = params['metric']
    print('user_dnn_hidden_units:{0},user_dnn_dropout:{1},out_dnn_activation:{2}'.format(params['user_dnn_hidden_units'],params['user_dnn_dropout'],params['out_dnn_activation']


))

    with tf.name_scope('user'):
        print('emb_columns:{0}'.format(user_columns))
        user_input_layer = tf.feature_column.input_layer(features=features, feature_columns=user_columns)
        print('user_input_layer:{0}'.format(user_input_layer))
        # user_output_layer = tf.layers.dense(inputs=user_input_layer, units=1, activation=None, use_bias=True)
        # print('emb_output_layer:{0}'.format(user_output_layer))

    with tf.name_scope('item'):
        item_input_layer = tf.feature_column.input_layer(features=features, feature_columns=item_columns)
        print('item_input_layer:{0}'.format(item_input_layer))
        # item_output_layer = tf.layers.dense(inputs=item_input_layer, units=1, activation=None, use_bias=True)
        # print('text_output_layer:{0}'.format(item_output_layer))

    # user tower
    for i in range(len(user_dnn_hidden_units)):
        if i == len(user_dnn_hidden_units) - 1:
            print('len:{0},{1}'.format(len(user_dnn_hidden_units),user_dnn_hidden_units[i]))
            user_dnn_out = CustomDense(units=user_dnn_hidden_units[i], dropout_rate=user_dnn_dropout[i],
                                       use_bn=dnn_use_bn, activation=out_dnn_activation, name='user_embed_out')(
                user_input_layer)
            break
        print('second len:{0},{1}'.format(len(user_dnn_hidden_units), user_dnn_hidden_units[i]))
        user_input_layer = CustomDense(units=user_dnn_hidden_units[i], dropout_rate=user_dnn_dropout[i],
                                       use_bn=dnn_use_bn, activation='relu', name='dnn_user_' + str(i))(
            user_input_layer)

    # item tower
    for i in range(len(item_dnn_hidden_units)):
        if i == len(item_dnn_hidden_units) - 1:
            item_dnn_out = CustomDense(units=item_dnn_hidden_units[i], dropout_rate=item_dnn_dropout[i],
                                       use_bn=dnn_use_bn, activation=out_dnn_activation, name='item_embed_out')(
                item_input_layer)
            break
        item_input_layer = CustomDense(units=item_dnn_hidden_units[i], dropout_rate=item_dnn_dropout[i],
                                       use_bn=dnn_use_bn, activation='relu', name='dnn_item_' + str(i))(
            item_input_layer)
    print('user_dnn_out:{0},item_dnn_out:{1}'.format(user_dnn_out,item_dnn_out))
    with tf.name_scope('logit'):
        # o_prob = tf.nn.sigmoid(o_layer)
        # predictions = tf.cast((o_prob > 0.5), tf.float32)
        score = Similarity(type_sim=metric, gamma=gamma)([user_dnn_out, item_dnn_out])
        output = tf.keras.layers.Activation("sigmoid", name="dssm_out")(score)
        predictions = tf.cast((output > 0.5), tf.float32)
    print('output:{0}'.format( output))
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': output,
            'label': predictions
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    labels = tf.reshape(labels, [-1, 1])
    print('labels:{0},output:{1}'.format(labels,output))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output))

    if mode == tf.estimator.ModeKeys.TRAIN:
        if params['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], beta1=0.9, beta2=0.999,
                                               epsilon=1e-8)
        elif params['optimizer'] == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'], initial_accumulator_value=1e-8)
        elif params['optimizer'] == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'], momentum=0.95)
        elif params['optimizer'] == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate=params['learning_rate'])
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        auc = tf.metrics.auc(labels, predictions)
        my_metrics = {
            'accuracy': tf.metrics.accuracy(labels, predictions),
            'auc': tf.metrics.auc(labels, predictions)
        }
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('auc', auc[1])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=my_metrics)

