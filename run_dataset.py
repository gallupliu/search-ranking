import os
import json
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as tf_text


# def _parse(line, vocab):
#     columns = tf.io.decode_csv(line, record_defaults=record_defaults, field_delim='\t')
#     parsed = dict(zip(COL_NAME, columns))
#     labels = parsed['label']
#     feature_dict = {}
#     def get_content(column):
#
#         tokens = tokenizer.tokenize(column)
#         cur_list_size = tokens.shape[0]
#         # print('first')
#         # print(tokens, cur_list_size)
#         ids = vocab.lookup(tokens)
#         ids = tf.reshape(ids, [1, -1])
#
#         # print('ids:{0} {1} {2}'.format(ids,ids.shape[0],tf.reshape(ids,[1,-1])))
#
#         def truncate_fn():
#             return tf.slice(ids, [0, 0], [batch_size, list_size])
#
#         def pad_fn():
#             return tf.pad(
#                 tensor=ids,
#                 paddings=[[0, 0], [0, list_size - cur_list_size]],
#                 constant_values=0)
#
#         ids = tf.cond(
#             pred=cur_list_size >= list_size, true_fn=truncate_fn, false_fn=pad_fn)
#
#         return ids
#     for column in TEXT_COLUMNS:
#         print(parsed[column])
#         ids = tf.py_function(get_content, [parsed[column]], [tf.int64, tf.int32])
#         feature_dict[column] = ids
#     for column in NUMERICAL_COLUMNS:
#         print('feat_name:{0},value:{1}'.format(column, parsed[column]))
#         feature_dict.update({column,parsed[column]})
#
#     for column in CATEGORY_COLUMNS:
#         print('feat_name:{0},value:{1}'.format(column, parsed[column]))
#         feature_dict.update({column,parsed[column]})
#
#     label = parsed['label']
#
#     return feature_dict,label

def test_data():
    hsy_data = {
        "keyword": ["安 慕 希", "牛 奶", "牛", "奶 粉", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
        "title": ["安 慕 希", "牛 奶", "牛", "奶 粉", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
        "brand": ["安 慕 希", "伊 利", "蒙 牛", "奶 粉", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
        "tag": ["酸 奶", "纯 牛 奶", "牛", "固 态 奶", "婴 儿 奶 粉", "液 态 奶", "牛 肉", "奶", "牛 肉 干", "牛 奶 口 味"],
        "volume": [1, 2, 3, 4, 5, 4.3, 1.2, 4.5, 1.0, 0.8],
        "type": [0, 1, 0, 1, 2, 1, 0, 0, 2, 1],
        "label": [0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    }
    hsy_df = pd.DataFrame(hsy_data)
    # print(hsy_df.head(10))
    hsy_df.to_csv('./milk.csv', index=False, sep='\t')


def get_item_embed(file_names, embedding_dim):
    item_bert_embed = []
    item_id = []
    for file in file_names:
        with open(file, 'r') as f:
            for line in f:
                feature_json = json.loads(line)
                # item_bert_embed.append(feature_json['post_id'])
                # item_id.append(feature_json['values'])
                for k, v in feature_json.items():
                    item_bert_embed.append(v)
                    item_id.append(k)

    print(len(item_id))
    item_id2idx = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=item_id,
            values=range(1, len(item_id) + 1),
            key_dtype=tf.string,
            value_dtype=tf.int32),
        default_value=0)
    item_bert_embed = [[0.0] * embedding_dim] + item_bert_embed
    item_embedding = tf.constant(item_bert_embed, dtype=tf.float32)
    return item_id2idx, item_embedding


def test_vocab():
    pets = {'pets': [['牛', '奶'], ['液', '体'], ['安', '慕'], ['婴', '儿'], ['口', '味']]}
    # pets = {"pets": ['牛','奶']}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fc_path = os.path.join(dir_path, 'char.txt')

    column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="pets",
        vocabulary_file=fc_path,
        num_oov_buckets=0)

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(pets, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))
        print(session.run([indicator]))


TEXT_COLUMNS = ["keyword", "title", "brand", "tag"]
NUMERICAL_COLUMNS = ["volume"]
CATEGORY_COLUMNS = ["type"]

COL_NAME = ["keyword", "title", "brand", "tag", "volume", "type", "label"]
record_defaults = [[""], [""], [""], [""], [0.0], [0], [0]]
embedding_dim = 32
char_file_names = ['./data/char.json']
CHAR_ID2IDX, CHAR_EMBEDDING = get_item_embed(char_file_names, embedding_dim)


def input_fn():
    # Create a lookup table for a vocabulary
    _VOCAB = ['[pad]', '<unk>', 'the', 'a', '牛', '奶', '液', '体', '安', '慕', '婴', '儿', '口', '味']

    _VOCAB_SIZE = len(_VOCAB)

    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=_VOCAB,
            key_dtype=tf.string,
            values=tf.range(
                tf.size(_VOCAB, out_type=tf.int64), dtype=tf.int64),
            value_dtype=tf.int64),
        num_oov_buckets=1
    )
    data_path = './milk.csv'
    # Extract lines from input files using the Dataset API.
    filenames = tf.data.Dataset.list_files([
        data_path,
    ])
    dataset = filenames.flat_map(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

    tokenizer = tf_text.WhitespaceTokenizer()

    batch_size = 2
    list_size = 5



    def _parse(example, vocab):
        columns = tf.io.decode_csv(example, record_defaults=record_defaults, field_delim='\t')
        parsed = dict(zip(COL_NAME, columns))

        feature_dict = {}
        for column in TEXT_COLUMNS:
            tokens_list = tf.strings.split([parsed[column]])
            print(tokens_list.values)
            print(tokens_list.to_tensor(default_value='', shape=[None, 10]))

            emb = tf.nn.embedding_lookup(params=CHAR_EMBEDDING, ids=CHAR_ID2IDX.lookup(tokens_list))
            # emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
            emb = tf.reduce_mean(emb, axis=0)
            print('first emb:{0}'.format(emb))
            feature_dict[column + '_emb'] = emb
            # tokens = tokenizer.tokenize(parsed['keyword'])
            tokens = tokens_list.to_tensor(default_value='', shape=[None, 10])
            cur_list_size = tokens.shape[0]
            print('first')
            print(tokens, cur_list_size)
            ids = vocab.lookup(tokens)
            feature_dict[column + '_ids'] = ids
            # ids = tf.reshape(ids, [1, -1])
            # print(ids.to_tensor(0))

            print('ids:{0} {1} {2}'.format(ids, ids.shape[0], tf.reshape(ids, [1, -1])))

        for column in NUMERICAL_COLUMNS:
            print('feat_name:{0},value:{1}'.format(column, parsed[column]))
            feature_dict[column] = parsed[column]

        for column in CATEGORY_COLUMNS:
            print('feat_name:{0},value:{1}'.format(column, parsed[column]))
            feature_dict[column] = parsed[column]
        labels = parsed['label']
        return feature_dict, labels

    tokenized_docs = dataset.map(lambda x: _parse(x, table))

    iterator = iter(tokenized_docs)
    print('value')
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))


if __name__ == "__main__":
    # test_data()

    input_fn()
