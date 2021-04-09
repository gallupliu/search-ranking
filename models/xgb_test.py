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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# 数据集并行化跑
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import LabelEncoder
import math
import warnings

print(pyspark.__version__)
os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_261.jdk/Contents/Home"
import json

warnings.filterwarnings('ignore')


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

parameters = {
    'max_depth': [5, 10, 15, 20, 25],
    # 'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    # 'n_estimators': [50, 100, 200, 300, 500],
    # 'min_child_weight': [0, 2, 5, 10, 20],
    # 'max_delta_step': [0, 0.2, 0.6, 1, 2],
    # 'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
    # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    # 'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
    # 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
    'scale_pos_weight': [1, 2]
}

xlf = xgb.XGBClassifier(max_depth=10,
                        learning_rate=0.01,
                        n_estimators=20,
                        objective='binary:logistic',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440,
                        eval_metric='auc',
                        missing=None)

# 有了gridsearch我们便不需要fit函数
# gsearch = GridSearchCV(xlf, param_grid=parameters,  cv=3)


gsearch = RandomizedSearchCV(estimator=xlf, param_distributions=parameters,
                               cv=5, n_iter=5, scoring='roc_auc', n_jobs=1, verbose=3, return_train_score=True,
                               random_state=121)
gsearch.fit(train_data, train_label)
print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    # 'metric': 'auc',
    'eval_metric': 'auc',
    'max_depth': 5,
    'min_child_weight': 350,
    'gamma': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'scale_pos_weight': 3,
}

watchlist = [(dtrain, 'train')]

bst = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
bst.get_score(importance_type='gain')
# 预测
ypred = bst.predict(dtest)

# 保存模型和加载模型
# bst.save_model('./xgb1.model')
# bst2 = xgb.core.Booster(model_file='./xgb1.model')
#
# s = sc.parallelize(test_data, 5)
#
# test_probs = clf.predict_proba(test_data)[:, 1]
