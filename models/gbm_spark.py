
from pyspark.sql import SparkSession
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression, GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline


# File location and type
file_location = "~/Downloads/creditcard.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

import pyspark
# spark = pyspark.sql.SparkSession.builder.appName("MyApp") \
#             .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1") \
#             .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
#             .getOrCreate()
# import mmlspark
spark = SparkSession.builder \
.appName("Churn Scoring LightGBM") \
.master("local[4]") \
.config("spark.jars", "/Users/gallup/study/search/RankServices/feature/lib/mmlspark_2.11-1.0.0-rc1.jar") \
.getOrCreate()
# from mmlspark.LightGBMRegressor import LightGBMRegressor
from mmlspark.lightgbm import LightGBMClassifier
# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

feature_cols = ["V" + str(i) for i in range(1,29)] + ["Amount"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
stages = [assembler]


best_params = {
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'eval_metric': 'binary_error',
    'feature_fraction': 0.944714847210862,
    'lambda_l1': 1.0,
    'lambda_l2': 45.0,
    'learning_rate': 0.1,
    'loss_function': 'binary_error',
    'max_bin': 60,
    'max_depth': 58,
    'metric': 'binary_error',
    'num_iterations': 379,
    'num_leaves': 850,
    'objective': 'binary',
    'random_state': 7,
    'verbose': None}

lgb = LightGBMClassifier(learningRate=0.1,
                  earlyStoppingRound=100,
                  featuresCol='features',
                  labelCol='Class',
                  isUnbalance=True,
                  baggingFraction=best_params["bagging_fraction"],
                  baggingFreq=1,
                  featureFraction=best_params["feature_fraction"],
                  lambdaL1=best_params["lambda_l1"],
                  lambdaL2=best_params["lambda_l2"],
                  maxBin=best_params["max_bin"],
                  maxDepth=best_params["max_depth"],
                  numIterations=best_params["num_iterations"],
                  numLeaves=best_params["num_leaves"],
                  objective="binary",
                  baggingSeed=7
                  )
stages += [lgb]

pipelineModel = Pipeline(stages=stages)

train, test = df.randomSplit([0.8, 0.2], seed=7)

model = pipelineModel.fit(train)

preds = model.transform(test)

binaryEvaluator = BinaryClassificationEvaluator(labelCol="Class")
print ("Test Area Under ROC: " + str(binaryEvaluator.evaluate(preds, {binaryEvaluator.metricName: "areaUnderROC"})))

tp = preds[(preds.Class == 1) & (preds.prediction == 1)].count()
tn = preds[(preds.Class == 0) & (preds.prediction == 0)].count()
fp = preds[(preds.Class == 0) & (preds.prediction == 1)].count()
fn = preds[(preds.Class == 1) & (preds.prediction == 0)].count()

print ("True Positives:", tp)

print ("True Negatives:", tn)

print ("False Positives:", fp)

print ("False Negatives:", fn)

print ("Total", preds.count())

r = float(tp)/(tp + fn)

print ("recall", r)

p = float(tp) / (tp + fp)

print ("precision", p)

f1 = 2 * p * r /(p + r)

print ("f1", f1)