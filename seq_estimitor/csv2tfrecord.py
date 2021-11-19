import csv
import tensorflow as tf

NUMERIC_FEATURE_NAMES = ['pax_price', 'driver_age', 'pax_age']
CATEGORICAL_FEATURE_NAMES = ['pax_des']
SEQ_FEATURE_NAMES = ['hist_price_id', 'hist_des_id']
TARGET_NAME = 'label'


def create_csv_iterator(csv_file_path, skip_header):
    with tf.gfile.Open(csv_file_path) as csv_file:
        reader = csv.reader(csv_file)
        if skip_header:  # Skip the header
            next(reader)
        for row in reader:
            yield row


def create_example(row, header):
    example = tf.train.Example()
    for i in range(len(header)):

        feature_name = header[i]
        feature_value = row[i]
        if feature_name in NUMERIC_FEATURE_NAMES:
            example.features.feature[feature_name].float_list.value.extend([float(feature_value)])

        if feature_name in SEQ_FEATURE_NAMES:
            example.features.feature[feature_name].float_list.value.extend(
                [float(value) for value in feature_value.replace('[', '').replace(']', '').split(',')])

        elif feature_name in CATEGORICAL_FEATURE_NAMES:
            example.features.feature[feature_name].bytes_list.value.extend([bytes(feature_value, 'utf-8')])
        elif feature_name in TARGET_NAME:
            example.features.feature[feature_name].float_list.value.extend([float(feature_value)])

    return example


def create_tfrecords_file(input_csv_file, header):
    output_tfrecord_file = input_csv_file.replace("csv", "tfrecords")
    writer = tf.python_io.TFRecordWriter(output_tfrecord_file)
    print("Creating TFRecords file at", output_tfrecord_file, "...")

    for i, row in enumerate(create_csv_iterator(input_csv_file, skip_header=True)):

        if len(row) == 0:
            continue

        example = create_example(row, header)
        content = example.SerializeToString()
        writer.write(content)

    writer.close()

    print("Finish Writing", output_tfrecord_file)


header = ['driver_price_seq', 'driver_des_seq', 'pax_des', 'pax_price', 'driver_age', 'pax_age', 'label']

create_tfrecords_file('./data/din_data.csv', header)


def input_fn_from_tfrecords(data_file, num_epochs, shuffle, batch_size):
    def _parse_TFRecords_fn(record):
        features = {
            'driver_price_seq': tf.io.FixedLenFeature([3], tf.float32),
            'driver_des_seq': tf.io.FixedLenFeature([3], tf.float32),
            'pax_des': tf.io.FixedLenFeature([], tf.string),
            'pax_price': tf.io.FixedLenFeature([], tf.float32),
            'driver_age': tf.io.FixedLenFeature([], tf.float32),
            'pax_age': tf.io.FixedLenFeature([], tf.float32),
            'label': tf.io.FixedLenFeature([], tf.float32)
        }
        features = tf.io.parse_single_example(record, features)
        labels = features.pop('label')
        return features, labels

    assert tf.io.gfile.exists(data_file), ('no file named: ' + str(data_file))

    dataset = tf.data.TFRecordDataset(data_file).map(_parse_TFRecords_fn, num_parallel_calls=10)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


train_dataset = input_fn_from_tfrecords('./data/din_data.tfrecords', 1, shuffle=True,
                                        batch_size=16)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())

    print('value')
    for i in range(1):
        print('dict:{0}'.format(train_dataset))
        print(session.run(train_dataset))
