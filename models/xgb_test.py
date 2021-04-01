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


#
# train = pd.read_csv("../input/train.csv")
# test = pd.read_csv("../input/test.csv")
#
#
# train = train.drop('QuoteNumber', axis=1)
# test = test.drop('QuoteNumber', axis=1)
#
# # Lets play with some dates
# train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
# train = train.drop('Original_Quote_Date', axis=1)
#
# test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
# test = test.drop('Original_Quote_Date', axis=1)
#
# train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
# train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
# train['weekday'] = train['Date'].dt.dayofweek
#
# test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
# test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
# test['weekday'] = test['Date'].dt.dayofweek
#
# train = train.drop('Date', axis=1)
# test = test.drop('Date', axis=1)
#
# #fill -999 to NAs
# train = train.fillna(-999)
# test = test.fillna(-999)
#
# features = list(train.columns[1:])  #la colonne 0 est le quote_conversionflag
# print(features)
#
#
# for f in train.columns:
#     if train[f].dtype=='object':
#         print(f)
#         lbl = preprocessing.LabelEncoder()
#         lbl.fit(list(train[f].values) + list(test[f].values))
#         train[f] = lbl.transform(list(train[f].values))
#         test[f] = lbl.transform(list(test[f].values))

xgb_model = xgb.XGBClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have
#much fun of fighting against overfit
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'scale_pos_weight': [1, 2],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}


clf = GridSearchCV(xgb_model, parameters, n_jobs=5,
                   cv=5,
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(train_data, train_label.stack().values.tolist())

#trust your CV!
means = clf.cv_results_['mean_test_score']
params = clf.cv_results_['params']
print('Raw AUC score:', means)
print('best score:{} with param:{}'.format(clf.best_score_, clf.best_params_))

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
bst.save_model('./xgb1.model')
bst2 = xgb.core.Booster(model_file='./xgb1.model')

s = sc.parallelize(test_data, 5)

test_probs = clf.predict_proba(test_data)[:,1]