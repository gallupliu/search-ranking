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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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

# A parameter grid for XGBoost
params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

clf = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic', nthread=4)

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

random_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4,
                                   cv=skf.split(train_data, train_label), verbose=3, random_state=1001)
random_search.fit(train_data, train_label)
print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('xgb-random-grid-search-results-01.csv', index=False)


# clf.fit(train_data, train_label.stack().values.tolist())
# print('best score:{} with param:{}'.format(clf.best_score_, clf.best_params_))

# xgboost模型参数
# params = {'booster': 'gbtree',
#           'objective': 'binary:logistic',
#           'eval_metric': 'auc',
#           'max_depth': 6,
#           'num_leaves': 63,
#           'lambda': 10,
#           'subsample': 0.75,
#           'colsample_bytree': 0.75,
#           'min_child_weight': 2,
#           'eta': 0.025,
#           'seed': 0,
#           'nthread': 8,
#           'silent': 1}

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'metric': 'auc',
    'eval_metric': 'auc',
    'max_depth': 5,
    'min_child_weight': 350,
    'gamma': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'scale_pos_weight': 3,
}
#
# watchlist = [(dtrain, 'train')]
#
# bst = xgb.train(params, dtrain, num_round=100, evals=watchlist)
# bst.get_score(importance_type='gain')
# # 预测
# ypred = bst.predict(dtest)
#
# # 保存模型和加载模型
# bst.save_model('./xgb2.model')
# bst2 = xgb.core.Booster(model_file='./xgb2.model')
#
# s = sc.parallelize(test_data, 5)

# 并行预测
# s.map(lambda x: bst2.predict(xgb.DMatrix(np.array(x).reshape((1, -1))))).collect()
