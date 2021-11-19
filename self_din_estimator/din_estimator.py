import tensorflow as tf
import numpy as np
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
from utils import embedding_dim, get_feature_col_type, get_seq_feature, sparse_string_join


def build_deep_layers(net, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.

    for layer_id, num_hidden_units in enumerate(params['hidden_units']):
        net = tf.layers.dense(net,
                              units=num_hidden_units,
                              # activation=tf.nn.relu,
                              activation=tf.nn.leaky_relu,
                              # activation=prelu,
                              # activation=dice,
                              kernel_initializer=tf.glorot_uniform_initializer())

    return net


def build_cross_layers(x0, params):
    num_layers = params['num_cross_layers']
    x = x0
    for i in range(num_layers):
        x = cross_layer(x0, x, 'cross_{}'.format(i))
    return x


def cross_layer(x0, x, name):
    with tf.variable_scope(name):
        input_dim = x0.get_shape().as_list()[1]
        w = tf.get_variable("weight", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("bias", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
        xb = tf.tensordot(tf.reshape(x, [-1, 1, input_dim]), w, 1)
        return x0 * xb + b + x


def attention_layer(querys, keys, keys_id):
    """
        queries:     [Batchsize, 1, embedding_size]
        keys:        [Batchsize, max_seq_len, embedding_size]  max_seq_len is the number of keys(e.g. number of clicked creativeid for each sample)
        keys_id:     [Batchsize, max_seq_len]
    """

    keys_length = tf.shape(keys)[1]  # padded_dim
    embedding_size = querys.get_shape().as_list()[-1]
    keys = tf.reshape(keys, shape=[-1, keys_length, embedding_size])
    querys = tf.reshape(tf.tile(querys, [1, keys_length]), shape=[-1, keys_length, embedding_size])

    net = tf.concat([keys, keys - querys, querys, keys * querys], axis=-1)
    for units in [32, 16]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.leaky_relu)
    att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)  # shape(batch_size, max_seq_len, 1)
    outputs = tf.reshape(att_wgt, shape=[-1, 1, keys_length], name="weight")  # shape(batch_size, 1, max_seq_len)

    scores = outputs
    scores = scores / (embedding_size ** 0.5)  # scale
    scores = tf.nn.softmax(scores)
    outputs = tf.matmul(scores, keys)  # (batch_size, 1, embedding_size)
    outputs = tf.reduce_sum(outputs, 1, name="attention_embedding")  # (batch_size, embedding_size)

    return outputs


def din_model_fn(features, labels, params, mode='train'):
    print('features:', features)
    print(params['feature_columns'])
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    print("mode=", mode)
    feature_conf_list = params['feature_conf_list']

    att_list = []
    for d in feature_conf_list:
        if d['feature_column_function'] == 'seq':
            seq_name = d['name']  # 'seq_item_id'
            # base_name=seq_name[4:] # 'item_id'
            base_name = d['base_col']  # 'item_id'
            hash_bucket_size = d['hash_bucket_size']
            base_col = features[base_name]
            seq_col = features[seq_name]
            base_feature = tf.string_to_hash_bucket_fast(base_col, hash_bucket_size)
            seq_feature = tf.string_to_hash_bucket_fast(seq_col, hash_bucket_size)
            var_name = base_name + "_embeddings"
            emb_size = embedding_dim(hash_bucket_size)
            seq_feature_embeddings = tf.get_variable(name=var_name, dtype=tf.float32,
                                                     shape=[hash_bucket_size, emb_size])

            base_emb = tf.nn.embedding_lookup(seq_feature_embeddings, base_feature)
            seq_emb = tf.nn.embedding_lookup(seq_feature_embeddings, seq_feature)
            seq_attention = attention_layer(base_emb, seq_emb, seq_feature)
            att_list.append(seq_attention)

    last_deep_layer = build_deep_layers(net, params)
    last_cross_layer = build_cross_layers(net, params)

    ll = [last_deep_layer, last_cross_layer] + att_list
    last_layer = tf.concat(ll, 1)
    head = head_lib._binary_logistic_or_multi_class_head(n_classes=2, weight_column=None,
                                                         label_vocabulary=None,
                                                         loss_reduction=losses.Reduction.SUM)
    logits = tf.layers.dense(last_layer, units=head.logits_dimension,
                             kernel_initializer=tf.glorot_uniform_initializer())

    if mode == tf.estimator.ModeKeys.PREDICT:
        preds = tf.sigmoid(logits, name='head/predictions/probabilities')
        # user_id = features['user_id']
        # label = features['label']
        predictions = {
            'probabilities': preds,
            # 'user_id': user_id,
            # 'label': label
        }
        export_outputs = {
            'regression': tf.estimator.export.PredictOutput(predictions['probabilities'])
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    label = tf.reshape(labels, [-1, 1])
    label = tf.cast(label, dtype=tf.float32)
    print('haha,mode=', mode)
    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=label,
        logits=logits,
        train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
    )


def build_estimator(model_dir, deep_columns, hidden_units, learning_rate, num_cross_layers, feature_conf_list, mode):
    model_config = tf.estimator.RunConfig().replace(
        save_checkpoints_steps=100,
        log_step_count_steps=1000,
        save_summary_steps=1000,
        keep_checkpoint_max=5)

    return tf.estimator.Estimator(
        model_fn=din_model_fn,
        model_dir=model_dir,
        params={
            'feature_columns': deep_columns,
            'hidden_units': hidden_units.split(','),
            'learning_rate': learning_rate,
            'num_cross_layers': num_cross_layers,
            'feature_conf_list': feature_conf_list,
            'mode': mode
        },
        config=model_config)


def input_fn_v5(data_file, num_epochs, shuffle, batch_size):
    col_feature, col_type, _ = get_feature_col_type()
    dataset = tf.data.experimental.make_csv_dataset(
        data_file,
        batch_size=batch_size,
        label_name='label',
        select_columns=col_feature,
        column_defaults=col_type,
        shuffle=shuffle,
        na_value="",
        num_epochs=num_epochs,
        num_parallel_reads=32,
        prefetch_buffer_size=4 * batch_size,
        ignore_errors=True)
    # https://stackoverflow.com/questions/48068013/how-to-speed-up-batch-preparation-when-using-estimators-api-combined-with-tf-dat
    seq_col = get_seq_feature()
    print('seq_col', seq_col)

    def dataset_fn(ds, label):
        print(ds['hist_order_name'])
        for key in seq_col:
            cols = tf.string_split(ds[key], delimiter=",")
            print(cols)
            cols = sparse_string_join(cols)
            print(cols)
            ds[key] = cols
        return ds, label

    # dataset = dataset.map(dataset_fn)
    dataset = dataset.map(dataset_fn, num_parallel_calls=32)
    dataset = dataset.make_one_shot_iterator().get_next()
    return dataset


def save_model(model, feature_columns, feature_conf_list, export_dir):
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    _, _, dict_col = get_feature_col_type()

    d_update = {}
    hist_embedding_len = 3
    for d in feature_conf_list:
        if d['feature_column_function'] == 'seq':
            seq_name = d['name']  # 'seq_item_id'
            d_type = dict_col[seq_name]
            if d_type == 'str':
                d_update[seq_name] = tf.FixedLenFeature(shape=[hist_embedding_len], dtype=tf.string)
            elif d_type == 'int64':
                d_update[seq_name] = tf.FixedLenFeature(shape=[hist_embedding_len], dtype=tf.int32)
            elif d_type == 'int64':
                d_update[seq_name] = tf.FixedLenFeature(shape=[hist_embedding_len], dtype=tf.int64)
            elif d_type == 'float64':
                d_update[seq_name] = tf.FixedLenFeature(shape=[hist_embedding_len], dtype=tf.float32)
            else:
                pass
            base_name = d['base_col']
            d_type = dict_col[base_name]
            if d_type == 'str':
                d_update[base_name] = tf.FixedLenFeature(shape=[], dtype=tf.string)
            elif d_type == 'int64':
                d_update[base_name] = tf.FixedLenFeature(shape=[], dtype=tf.int32)
            elif d_type == 'int64':
                d_update[base_name] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
            elif d_type == 'float64':
                d_update[base_name] = tf.FixedLenFeature(shape=[], dtype=tf.float32)
            else:
                pass

    feature_spec.update(d_update)

    print(feature_spec)
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    print(export_input_fn)

    model.export_savedmodel(export_dir, export_input_fn)
