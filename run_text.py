# import tensorflow as tf
#
# import tensorflow_datasets as tfds
# import os
#
# DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
# FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']
#
# for name in FILE_NAMES:
#     text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name)
#
# parent_dir = os.path.dirname(text_dir)
#
# def labeler(example, index):
#   return example, tf.cast(index, tf.int64)
#
# labeled_data_sets = []
#
# for i, file_name in enumerate(FILE_NAMES):
#   lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
#   labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
#   labeled_data_sets.append(labeled_dataset)
#
#
# BUFFER_SIZE = 50000
# BATCH_SIZE = 64
# TAKE_SIZE = 5000
#
# all_labeled_data = labeled_data_sets[0]
# for labeled_dataset in labeled_data_sets[1:]:
#     all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
#
# all_labeled_data = all_labeled_data.shuffle(
#     BUFFER_SIZE, reshuffle_each_iteration=False)
#
# for ex in all_labeled_data.take(5):
#   print(ex)
#
# tokenizer = tfds.deprecated.text.Tokenizer()
#
# vocabulary_set = set()
# for text_tensor, _ in all_labeled_data:
#   some_tokens = tokenizer.tokenize(text_tensor.numpy())
#   vocabulary_set.update(some_tokens)
#
# vocab_size = len(vocabulary_set)
#
#
# encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set)
#
#
# example_text = next(iter(all_labeled_data))[0].numpy()
# print(example_text)
#
# encoded_example = encoder.encode(example_text)
# print(encoded_example)
#
#
# def encode(text_tensor, label):
#   encoded_text = encoder.encode(text_tensor.numpy())
#   return encoded_text, label
#
# def encode_map_fn(text, label):
#   # py_func doesn't set the shape of the returned tensors.
#   encoded_text, label = tf.py_function(encode,
#                                        inp=[text, label],
#                                        Tout=(tf.int64, tf.int64))
#
#   # `tf.data.Datasets` work best if all components have a shape set
#   #  so set the shapes manually:
#   encoded_text.set_shape([None])
#   label.set_shape([])
#
#   return encoded_text, label
#
#
# all_encoded_data = all_labeled_data.map(encode_map_fn)
#
#
#
# train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
# train_data = train_data.padded_batch(BATCH_SIZE)
#
# test_data = all_encoded_data.take(TAKE_SIZE)
# test_data = test_data.padded_batch(BATCH_SIZE)
#
#
# sample_text, sample_labels = next(iter(test_data))
#
# sample_text[0], sample_labels[0]
#
# vocab_size += 1
#
# model = tf.keras.Sequential()
#
# model.add(tf.keras.layers.Embedding(vocab_size, 64))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
#
# # 一个或多个紧密连接的层
# # 编辑 `for` 行的列表去检测层的大小
# for units in [64, 64]:
#   model.add(tf.keras.layers.Dense(units, activation='relu'))
#
# # 输出层。第一个参数是标签个数。
# model.add(tf.keras.layers.Dense(3, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_data, epochs=3, validation_data=test_data)
# eval_loss, eval_acc = model.evaluate(test_data)
#
# print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
#


import tensorflow as tf
import tensorflow_text as text
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())

docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'],
                                           ["It's a trap!"]])
tokenizer = text.WhitespaceTokenizer()
tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))
iterator = tf.compat.v1.data.make_one_shot_iterator(tokenized_docs)
print(iterator.get_next().to_list())
print(iterator.get_next().to_list())

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=200, split=" ")
sample_text = ["This is a sample sentence1 created by sample person AB.CDEFGHIJKLMNOPQRSTUVWXYZ",
               "This is another sample sentence1 created by another sample person AB.CDEFGHIJKLMNOPQRSTUVWXYZ"]

tokenizer.fit_on_texts(sample_text)

print (tokenizer.texts_to_sequences(["sample person AB.CDEFGHIJKLMNOPQRSTUVWXYZ"]))

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def read_dataset(prefix, mode, batch_size):

    def _input_fn():

        def decode_csv(value_column):

            columns = tf.decode_csv(value_column, field_delim='|', record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))

            features['comment_words'] = tf.string_split([features['comment']])
            features['comment_words'] = tf.sparse_tensor_to_dense(features['comment_words'], default_value=PADWORD)
            features['comment_padding'] = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
            features['comment_padded'] = tf.pad(features['comment_words'], features['comment_padding'])
            features['comment_sliced'] = tf.slice(features['comment_padded'], [0,0], [-1, MAX_DOCUMENT_LENGTH])
            features['comment_words'] = tf.pad(features['comment_sliced'], features['comment_padding'])
            features['comment_words'] = tf.slice(features['comment_words'],[0,0],[-1,MAX_DOCUMENT_LENGTH])

            features.pop('comment_padding')
            features.pop('comment_padded')
            features.pop('comment_sliced')

            label = features.pop(LABEL_COLUMN)

            return features, label

        # Use prefix to create file path
        file_path = '{}/{}*{}*'.format(INPUT_DIR, prefix, PATTERN)

        # Create list of files that match pattern
        file_list = tf.gfile.Glob(file_path)

        # Create dataset from file list
        dataset = (tf.data.TextLineDataset(file_list)  # Read text file
                    .map(decode_csv))  # Transform each elem by applying decode_csv fn

        tf.logging.info("...dataset.output_types={}".format(dataset.output_types))
        tf.logging.info("...dataset.output_shapes={}".format(dataset.output_shapes))

        if mode == tf.estimator.ModeKeys.TRAIN:

            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)

        else:

            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)

        return dataset.make_one_shot_iterator().get_next()

    return _input_fn

# Define feature columns
def get_wide_deep():

    EMBEDDING_SIZE = 10

    # Define column types
    subreddit = tf.feature_column.categorical_column_with_vocabulary_list('subreddit', ['news', 'ireland', 'pics'])

    comment_embeds = tf.feature_column.embedding_column(
        categorical_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key='comment_words',
            vocabulary_file='{}/vocab.csv-00000-of-00001'.format(INPUT_DIR),
            vocabulary_size=100
            ),
        dimension = EMBEDDING_SIZE
        )

    # Sparse columns are wide, have a linear relationship with the output
    wide = [ subreddit ]

    # Continuous columns are deep, have a complex relationship with the output
    deep = [ comment_embeds ]

    return wide, deep