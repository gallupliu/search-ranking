# import tensorflow as tf
import tensorflow.compat.v1 as tf



def deepfm_model_fn(features, labels, mode, params):
    deep_columns     = params['deep_columns']
    deep_fields_size = params['deep_fields_size']
    org_emb_size     = params['embedding_dim']
    wide_columns     = params['wide_columns']
    wide_fields_size = params['wide_fields_size']
    emb_columns = params['emb_columns']
    emb_fields_size = params['emb_fields_size']
    text_columns = params['text_columns']
    text_fields_size = params['text_fields_size']

    print('deep_fields_size:{0} org_emb_size:{1}'.format(deep_fields_size, org_emb_size))
    print('features:{0} '.format(features))
    print('deep_column:{0}'.format( deep_columns))
    print('wide_columns:{0}'.format(wide_columns))
    deep_input_layer = tf.feature_column.input_layer(features=features,feature_columns=deep_columns)
    print('deep_input_layer:{0}'.format(deep_input_layer))

    with tf.name_scope('emb'):
        print('emb_columns:{0}'.format(emb_columns))
        emb_input_layer = tf.feature_column.input_layer(features=features, feature_columns=emb_columns)
        print('emb_input_layer:{0}'.format(emb_input_layer))
        emb_output_layer = tf.layers.dense(inputs=emb_input_layer, units=1, activation=None, use_bias=True)
        print('emb_output_layer:{0}'.format(emb_output_layer))

    with tf.name_scope('text'):
        text_input_layer = tf.feature_column.input_layer(features=features, feature_columns=text_columns)
        print('text_input_layer:{0}'.format(text_input_layer))
        text_output_layer = tf.layers.dense(inputs=text_input_layer, units=1, activation=None, use_bias=True)
        print('text_output_layer:{0}'.format(text_output_layer))

    with tf.name_scope('wide'):
        wide_input_layer = tf.feature_column.input_layer(features=features, feature_columns=wide_columns)
        print('wide_input_layer:{0}'.format(wide_input_layer))
        wide_output_layer = tf.layers.dense(inputs=wide_input_layer, units=1, activation=None, use_bias=True)
        print('wide_output_layer:{0}'.format(wide_output_layer))

    with tf.name_scope('deep'):
        d_layer_1 = tf.layers.dense(inputs=deep_input_layer, units=50, activation=tf.nn.relu, use_bias=True)
        bn_layer_1 = tf.layers.batch_normalization(inputs=d_layer_1, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
        deep_output_layer = tf.layers.dense(inputs=bn_layer_1, units=40, activation=tf.nn.relu, use_bias=True)
        print('deep_output_layer:{0}'.format(deep_output_layer))

    with tf.name_scope('fm'):
        total_feat = tf.reshape(deep_input_layer, [-1, deep_fields_size, org_emb_size])
        sum_square_part = tf.square(tf.reduce_sum(total_feat, 1))
        # print('sum_square_part:', sum_square_part.get_shape().as_list())
        square_sum_part = tf.reduce_sum(tf.square(total_feat), 1)
        # print('square_sum_part:', square_sum_part.get_shape().as_list())
        second_order_part = 0.5 * tf.subtract(sum_square_part, square_sum_part)
        # print('second_order_part:', second_order_part.get_shape().as_list())

    with tf.name_scope('concat'):
        # m_layer = tf.concat([wide_output_layer, deep_output_layer, second_order_part], 1)
        m_layer = tf.concat([wide_output_layer, deep_output_layer,emb_output_layer, text_output_layer,second_order_part], 1)
        o_layer = tf.layers.dense(inputs=m_layer, units=1, activation=None, use_bias=True)

    with tf.name_scope('logit'):
        o_prob = tf.nn.sigmoid(o_layer)
        predictions = tf.cast((o_prob > 0.5), tf.float32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities':    o_prob,
            'label':            predictions
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    labels = tf.reshape(labels, [-1,1])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=o_layer))

    if mode == tf.estimator.ModeKeys.TRAIN:
        if   params['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], beta1=0.9, beta2=0.999, epsilon=1e-8)
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
            'auc':      tf.metrics.auc(labels,predictions)
        }
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('auc', auc[1])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=my_metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=columns)
        print('feature_spec:{0}'.format(feature_spec))
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel(EXPORT_PATH, serving_input_fn)

