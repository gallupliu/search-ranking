import functools
import pandas as pd
import numpy as np
import tensorflow as tf

# 一、加载数据
# 网上的文件路径。数据为泰坦尼克号的旅客的生存数据。
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
# 下载到本地后的文件路径
# train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
train_file_path = './hys_df_test.csv'
test_file_path = './hys_df_test.csv'

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
# 指定标签列及其取值
TEXT_FEATURES = ['keyword', 'title', 'brand', 'tag']
NUMERICAL_FEATURES = ['volume']
CATEGORY_FEATURES = ['type']
LABEL_COLUMN = 'label'
LABELS = [0, 1]


# 这里使用tensorflow自带的加载器tf.data.experimental.make_csv_dataset读取CSV数据并创建数据集
# 另外我们也可以使用pandan来加载数据
def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=2,  # Artificially small to make examples easier to show.一批五个样本
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        field_delim='\t',
        **kwargs)
    return dataset


# 下面两行此时还未获取数据，get_dataset是一个懒加载函数，需要执行take()才开始获取数据，
# 获取后的数据集中的每项为一批（这里一批是5个样本），它们按列组成张量被组合在一起（例如性别为一列，年级为一列）。
raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)


# 二、数据预处理
# 由于CSV文件往往包含不同类型的数据，所以我们需要对不同类型的数据分别处理然后再合并。这里我们使用tensorflow自带的tf.feature_column来处理。
# 1、对于连续型数值数据的处理举例如下：
# 1.1定义一个预处理器用来仅选择数值列并将数值列打包入一个单独的列中
class Features(object):
    def __init__(self, text_features, numerical_features, category_features):
        self.text_features = text_features
        self.numerical_features = numerical_features
        self.category_features = category_features

    def __call__(self, features, labels):
        print('features:{0} {1}'.format(type(features['title']), features['title']))
        features['numeric'] = self.process_numeric_feature(features)
        features = self.process_text_feature(features)

        return features, labels

    def process_numeric_feature(self, features):
        numeric_features = [features.pop(name) for name in self.numerical_features]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        return numeric_features

    def process_text_feature(self, features):
        cur_list_size = tf.shape(input=serialized_list)[1]
        def truncate_fn():
            return tf.slice(serialized_list, [0, 0], [batch_size, list_size])

        def pad_fn():
            return tf.pad(
                tensor=serialized_list,
                paddings=[[0, 0], [0, list_size - cur_list_size]],
                constant_values="")

        serialized_list = tf.cond(
            pred=cur_list_size > list_size, true_fn=truncate_fn, false_fn=pad_fn)

        for feat_name in self.text_features:
            tokens = tf.strings.split(features[feat_name]
                                      )
            features[feat_name + '_char'] = tokens
        return features




packed_train_data = raw_train_data.map(
    Features(TEXT_FEATURES, NUMERICAL_FEATURES, CATEGORY_FEATURES))

packed_test_data = raw_test_data.map(
    Features(TEXT_FEATURES, NUMERICAL_FEATURES, CATEGORY_FEATURES))


# 定义一个show_batch函数来看处理的结果
def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("key:{:20s}: value{}".format(key, value.numpy()))


show_batch(packed_train_data)

# 1.2 对连续型数值数据进行归一化处理
# 连续型数值数据应该要进行归一化处理，由于归一化处理涉及到均值和方差，所以我们可以如下使用pandas来得到均值和方差：
data = pd.read_csv(train_file_path, sep='\t')[TEXT_FEATURES]
print(data)
desc = pd.read_csv(train_file_path, sep='\t')[NUMERICAL_FEATURES].describe()
print(desc)

# 取出均值和方差
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])


# 定义归一化函数
def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data - mean) / std


# 创建一个对数据进行归一化的数值列结构
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERICAL_FEATURES)])
numeric_columns = [numeric_column]

# 创建一个数值层
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)

# 2、分类数据的处理
# CSV文件中往往还包含分类数据，我们使用tf.feature_column来进行处理，使用tf.feature_column.indicator_column来人每个分类列创建一个集合：
# 定义一个分类字典
CATEGORIES = {
    'type': [0, 1, 2]
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

text_tokens = []
text_columns = []
char_vocabulary_file = './char.txt'
char_vocabulary_size = 19
for feat_nmae in TEXT_FEATURES:
    feature = tf.feature_column.categorical_column_with_vocabulary_file(
        feat_nmae + '_char', char_vocabulary_file, char_vocabulary_size)
    text_column = tf.feature_column.embedding_column(
        feature, 32, combiner='mean', initializer=None,
        ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True,
        use_safe_embedding_lookup=True
    )
    text_columns.append(text_column)
    # text_tokens.append(feature)

# text_columns = tf.feature_column.shared_embeddings(
#     text_tokens, dimension=32)

# 创建一个分类层
categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
text_layer = tf.keras.layers.DenseFeatures(text_columns)
# 合并处理层。该层提取并处理这两种类型数据
preprocessing_layer = tf.keras.layers.DenseFeatures(text_columns + categorical_columns + numeric_columns)

# 三、使用tf.keras.Sequential建造一个线性神经网络模型

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

# 四、训练、评估和预测
# 训练数据和测试数据
train_data = packed_train_data.shuffle(500)
test_data = packed_test_data
# 训练
model.fit(train_data, epochs=20)
# 在测试数据在进行评估
test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
# 预测
predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    prediction = tf.sigmoid(prediction).numpy()
    print("Predicted survival: {:.2%}".format(prediction[0]),
          " | Actual outcome: ",
          ("SURVIVED" if bool(survived) else "DIED"))
