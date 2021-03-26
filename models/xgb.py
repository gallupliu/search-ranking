import os
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import numpy as np
from pyspark.sql import SparkSession
# 构建xgboost模型
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# 数据集并行化跑
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import LabelEncoder
import math

print(pyspark.__version__)
os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_261.jdk/Contents/Home"
import json


def create_spark():
    spark_conf = SparkConf().setAppName("test").set("spark.ui.showConsoleProgress", "false") \
        .set("spark.driver.maxResultSize", "4g") \
        .set("spark.driver.memory", "4g") \
        .set("spark.executor.memory", "4g")

    sc = SparkContext(conf=spark_conf)
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
    return sc, spark


def encodeColumns(sdf, colnames):
    df = sdf  # .copy()
    labelEncoderDict = {}
    for col in colnames:
        print(col)
        labelEncoderDict[col] = {}
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        clas = le.classes_
        for i in range(0, len(clas)):
            labelEncoderDict[col][clas[i]] = i

    return df, labelEncoderDict


## 导出线上文件

def key_to_json(data):
    if data is None or isinstance(data, (bool, int, str, float)):
        return data
    if isinstance(data, (tuple, frozenset)):
        return str(data)
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.float):
        return int(data)
    raise TypeError


def to_json(data):
    if data is None or isinstance(data, (bool, int, tuple, range, str, list)):
        return data
    if isinstance(data, (set, frozenset)):
        return sorted(data)
    if isinstance(data, np.float):
        return float(data)
    if isinstance(data, dict):
        return {key_to_json(key): to_json(data[key]) for key in data}
    raise TypeError


class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self


def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in
                    zip(Y_pred, y_true)) / len(Y_pred)


sc, spark = create_spark()

df = spark.read.format("csv").option("header", "true").option("delimiter", ",").load(
    "/Users/gallup/study/search/rank/RankService/feature/src/main/scala/com/example/feature/hour.csv")
df.show()
features_col = ["temp", "atemp", "hum", "casual", "cnt", "season", "yr", "mnth", "hr"]
label_col = ["workingday"]
df = df.toPandas()
df, le_dict = encodeColumns(df, label_col)
train_df, test_df = train_test_split(df, test_size=0.2)

train_data = pd.DataFrame(train_df, columns=features_col)
train_label = pd.DataFrame(train_df, columns=label_col)
test_data = pd.DataFrame(test_df, columns=features_col)
test_label = pd.DataFrame(test_df, columns=label_col)

# label encoder dict
with open(f'''./label_encoder_dict.json''', 'w') as fp:
    json.dump(to_json(le_dict), fp)

for feat in features_col:
    train_data[feat] = pd.to_numeric(train_data[feat], downcast='float')
    test_data[feat] = pd.to_numeric(test_data[feat], downcast='float')

dtrain = xgb.DMatrix(train_data, label=train_label)
dtest = xgb.DMatrix(test_data)

clf = XGBoostClassifier(
    eval_metric='auc',
    num_class=2,
    nthread=4,
)
# parameters = {
#     'num_boost_round': [100, 250, 500],
#     'eta': [0.05, 0.1, 0.3],
#     'max_depth': [6, 9, 12],
#     'subsample': [0.9, 1.0],
#     'colsample_bytree': [0.9, 1.0],
# }
parameters = {
    # 'num_boost_round': [100, 250, 500],
    # 'eta': [0.05, 0.1, 0.3],
    # 'max_depth': [5,6],
    # 'subsample': [0.9, 1.0],
    # 'colsample_bytree': [0.9, 1.0],
    'scale_pos_weight':[1,2]
}
clf = GridSearchCV(clf, parameters, n_jobs=1, cv=2)

clf.fit(train_data, train_label.stack().values.tolist())
print('best score:{} with param:{}'.format(clf.best_score_, clf.best_params_))
print('predicted:', clf.predict([[1, 1]]))

# xgboost模型参数
params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'max_depth': 6,
          'num_leaves': 63,
          'lambda': 10,
          'subsample': 0.75,
          'colsample_bytree': 0.75,
          'min_child_weight': 2,
          'eta': 0.025,
          'seed': 0,
          'nthread': 8,
          'silent': 1}

watchlist = [(dtrain, 'train')]

bst = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
bst.get_score(importance_type='gain')
# 预测
ypred = bst.predict(dtest)

# 保存模型和加载模型
bst.save_model('./xgb2.model')
bst2 = xgb.core.Booster(model_file='./xgb2.model')

s = sc.parallelize(test_data, 5)

# 并行预测
# s.map(lambda x: bst2.predict(xgb.DMatrix(np.array(x).reshape((1, -1))))).collect()
