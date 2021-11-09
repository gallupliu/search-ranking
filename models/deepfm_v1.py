# import tensorflow as tf
import tensorflow.compat.v1 as tf


def deepfm_model_fn(features, labels, mode, params):
    deep_columns = params['deep_columns']
    deep_fields_size = params['deep_fields_size']
    org_emb_size = params['embedding_dim']
    wide_columns = params['wide_columns']
    wide_fields_size = params['wide_fields_size']
    emb_columns = params['emb_columns']
    emb_fields_size = params['emb_fields_size']
    text_columns = params['text_columns']
    text_fields_size = params['text_fields_size']

    print('deep_fields_size:{0} org_emb_size:{1}'.format(deep_fields_size, org_emb_size))
    print('features:{0} '.format(features))
    print('labels:{0} '.format(labels))
    print('deep_column:{0}'.format(deep_columns))
    print('wide_columns:{0}'.format(wide_columns))
    deep_input_layer = tf.feature_column.input_layer(features=features, feature_columns=deep_columns)
    print('deep_input_layer:{0}'.format(deep_input_layer))

    with tf.name_scope('emb'):
        print('emb_columns:{0}'.format(emb_columns))
        emb_input_layer = tf.feature_column.input_layer(features=features, feature_columns=emb_columns)

        print('emb_input_layer:{0}'.format(emb_input_layer))
        emb_output_layer = tf.layers.dense(inputs=emb_input_layer, units=1, activation=None, use_bias=True)
        print('emb_output_layer:{0}'.format(emb_output_layer))

    with tf.name_scope('text'):
        text_outputs = []
        for column in text_columns:
            sequence_feature_layer = tf.keras.experimental.SequenceFeatures(column)
            text_input_layer, sequence_length = sequence_feature_layer(features)
            sequence_length_mask = tf.sequence_mask(sequence_length)
            # text_input_layer = tf.feature_column.input_layer(features=features, feature_columns=text_columns)
            print('text_input_layer:{0},sequence_length:{1},sequence_length_mask:{2}'.format(text_input_layer,
                                                                                             sequence_length,
                                                                                             sequence_length_mask))
            if params['module'] == 'cnn':
                pooled_outputs = []
                text_feat = tf.reshape(text_input_layer, [-1, params['sequence_length'], params['embedding_size']])
                print('text_feat:{0}'.format(text_feat))
                text_feat = tf.expand_dims(text_feat, -1)
                with tf.name_scope("TextCNN"):
                    for i, filter_size in enumerate(params['filter_sizes']):
                        with tf.name_scope("conv-maxpool-%s" % filter_size):
                            # Convolution Layer
                            filter_shape = [filter_size, params['embedding_size'], 1, params['num_filters']]
                            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), dtype=tf.float32, name="W")
                            b = tf.Variable(tf.constant(0.1, shape=[params['num_filters']]), dtype=tf.float32, name="b")
                            conv = tf.nn.conv2d(
                                text_feat,
                                W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")
                            # Apply nonlinearity
                            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                            # Maxpooling over the outputs
                            pooled = tf.nn.max_pool(
                                h,
                                ksize=[1, params['sequence_length'] - filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name="pool")
                            pooled_outputs.append(pooled)
                # Combine all the pooled features
                num_filters_total = params['num_filters'] * len(params['filter_sizes'])
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
                # Dropout
                text_output_layer = tf.nn.dropout(h_pool_flat, keep_prob=params['keep_prob'])
                text_output_layer = tf.layers.dense(inputs=text_output_layer, units=50, activation=None, use_bias=True)
            else:
                text_output_layer = tf.layers.dense(inputs=text_input_layer, units=1, activation=None, use_bias=True)

            print('text_output_layer:{0}'.format(text_output_layer))
            text_outputs = text_output_layer

    with tf.name_scope('wide'):
        wide_input_layer = tf.feature_column.input_layer(features=features, feature_columns=wide_columns)
        print('wide_input_layer:{0}'.format(wide_input_layer))
        wide_output_layer = tf.compat.v1.layers.dense(inputs=wide_input_layer, units=1, activation=None, use_bias=True)
        print('wide_output_layer:{0}'.format(wide_output_layer))

    with tf.name_scope('deep'):
        d_layer_1 = tf.compat.v1.layers.dense(inputs=deep_input_layer, units=50, activation=tf.nn.relu, use_bias=True)
        bn_layer_1 = tf.layers.batch_normalization(inputs=d_layer_1, axis=-1, momentum=0.99, epsilon=0.001, center=True,
                                                   scale=True)
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
        print('wide_output_layer:{0}'.format(wide_output_layer))
        print('deep_output_layer:{0}'.format(deep_output_layer))
        print('text_outputs:{0}'.format(text_outputs))
        # text_outputs = tf.reshape(text_outputs, [params['batch_size'], -1])
        print('text_outputs_1:{0}'.format(tf.reshape(text_outputs, [params['batch_size'],  -1])))
        print('second_order_part:{0}'.format(second_order_part))
        m_layer = tf.concat(
            [wide_output_layer, deep_output_layer, text_outputs, second_order_part], 1)

        print('m_layer:{0}'.format(m_layer))
        o_layer = tf.layers.dense(inputs=m_layer, units=1, activation=None, use_bias=True)
        print('o_layer:{0}'.format(o_layer))

    with tf.name_scope('logit'):
        o_prob = tf.nn.sigmoid(o_layer)
        predictions = tf.cast((o_prob > 0.5), tf.float32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': o_prob,
            'label': predictions
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    print('labels reshape:{0} '.format(labels))
    labels = tf.reshape(labels, [-1, 1])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=o_layer))
    print('labels reshape 1:{0} '.format(labels))
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
        print('labels reshape acc:{0} '.format(labels))
        accuracy = tf.metrics.accuracy(labels, predictions)
        auc = tf.metrics.auc(labels, predictions)
        print('labels reshape auc:{0} '.format(labels))
        my_metrics = {
            'accuracy': tf.metrics.accuracy(labels, predictions),
            'auc': tf.metrics.auc(labels, predictions)
        }
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('auc', auc[1])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=my_metrics)

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=columns)
    #     print('feature_spec:{0}'.format(feature_spec))
    #     serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    #     model.export_savedmodel(EXPORT_PATH, serving_input_fn)
