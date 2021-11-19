import tensorflow as tf

from deepctr.estimator.feature_column import input_from_feature_columns, is_embedding
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features, get_linear_logit
from deepctr.inputs import create_embedding_matrix, embedding_lookup, get_dense_input, varlen_embedding_lookup, \
    get_varlen_pooling_list
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.sequence import AttentionSequencePoolingLayer
from deepctr.estimator.utils import deepctr_model_fn, variable_scope

from deepctr.layers.utils import concat_func, NoMask, combined_dnn_input, add_func

LINEAR_SCOPE_NAME = 'linear'
DNN_SCOPE_NAME = 'dnn'


def DINEstimator(dnn_feature_columns, history_feature_list, dnn_use_bn=False,
                 dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
                 att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024,
                 task='binary', model_dir=None, config=None, linear_optimizer='Ftrl',
                 dnn_optimizer='Adagrad', training_chief_hooks=None):
    def _model_fn(features, labels, mode, config):
        train_flag = (mode == tf.estimator.ModeKeys.TRAIN)
        with variable_scope(DNN_SCOPE_NAME):
            sparse_feature_columns = []
            dense_feature_columns = []
            varlen_sparse_feature_columns = []

            for feat in dnn_feature_columns:

                new_feat_name = list(feat.parse_example_spec.keys())[0]
                if new_feat_name in ['hist_price_id', 'hist_des_id']:
                    varlen_sparse_feature_columns.append(
                        VarLenSparseFeat(
                            SparseFeat(new_feat_name, vocabulary_size=100, embedding_dim=32, use_hash=False), maxlen=3)
                    )
                elif is_embedding(feat):
                    sparse_feature_columns.append(
                        SparseFeat(new_feat_name, vocabulary_size=feat[0]._num_buckets + 1,
                                   embedding_dim=feat.dimension))
                else:
                    dense_feature_columns.append(DenseFeat(new_feat_name))

            history_feature_columns = []
            sparse_varlen_feature_columns = []
            history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
            for fc in varlen_sparse_feature_columns:
                feature_name = fc.name
                if feature_name in history_fc_names:
                    history_feature_columns.append(fc)
                else:
                    sparse_varlen_feature_columns.append(fc)
            my_feature_columns = sparse_feature_columns + dense_feature_columns + varlen_sparse_feature_columns
            embedding_dict = create_embedding_matrix(my_feature_columns, l2_reg_embedding, seed, prefix="")

            query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, history_feature_list,
                                              history_feature_list, to_list=True)
            print('query_emb_list', query_emb_list)
            print('embedding_dict', embedding_dict)
            print('haha')
            print('history_feature_columns', history_feature_columns)
            print('haha')
            keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names,
                                             history_fc_names, to_list=True)
            print('keys_emb_list', keys_emb_list)
            dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                                  mask_feat_list=history_feature_list, to_list=True)
            print('dnn_input_emb_list', dnn_input_emb_list)
            dense_value_list = get_dense_input(features, dense_feature_columns)
            sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
            sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                          to_list=True)

            dnn_input_emb_list += sequence_embed_list

            keys_emb = concat_func(keys_emb_list, mask=True)
            deep_input_emb = concat_func(dnn_input_emb_list)
            query_emb = concat_func(query_emb_list, mask=True)
            hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                                 weight_normalization=att_weight_normalization, supports_masking=True)([
                query_emb, keys_emb])

            deep_input_emb = tf.keras.layers.Concatenate()([NoMask()(deep_input_emb), hist])
            deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
            dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
            output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
            final_logit = tf.keras.layers.Dense(1, use_bias=False,
                                                kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)
        #             logits_list.append(final_logit)
        #         logits = add_func(logits_list)
        #             print(labels)
        #             tf.summary.histogram(final_logit + '/final_logit', final_logit)
        return deepctr_model_fn(features, mode, final_logit, labels, task, linear_optimizer, dnn_optimizer,
                                training_chief_hooks=training_chief_hooks)

    return tf.estimator.Estimator(_model_fn, model_dir=model_dir, config=config)
