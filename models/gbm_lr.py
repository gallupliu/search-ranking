import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

#------path --------
path = './data/'
path_train_03 =path+ 'clean_data/clean_data2019-03-31.csv'
path_train_04 =path+ 'clean_data/clean_data2019-04-28.csv'
path_train_05 =path+ 'clean_data/clean_data2019-06-02.csv'

#--------goal
path_this_month_0331 ='./data/目标函数/2019-01-01_2019-03-31_goal.csv'
path_this_month_0428 ='./data/目标函数/2019-02-11_2019-04-28_goal.csv'
path_this_month_0602 ='./data/目标函数/2019-04-25_2019-06-02_goal.csv'

path_goal_0331_next_month ='./data/目标函数/next_month_2019-03-31_2019-04-30_goal.csv'
path_goal_0428_next_month ='./data/目标函数/next_month_2019-04-28_2019-05-31_goal.csv'
path_goal_0602_next_month ='./data/目标函数/next_month_2019-06-02_2019-06-30_goal.csv'


#---------fund-------------
path_fund_score_stock = path+'基金标签/用户基金评分股票.csv'
path_fund_score_no_stock = path+'基金标签/用户基金评分非股票.csv'
#---------fund score--------
path_fund_score_03 = path+'用户对基金的评分（购买和浏览）/pre_deal_feature_data_2019-03-31_total.csv'
path_fund_score_04 = path+'用户对基金的评分（购买和浏览）/pre_deal_feature_data_2019-04-28_total.csv'
path_fund_score_05 = path+'用户对基金的评分（购买和浏览）/pre_deal_feature_data_2019-06-02_total.csv'


df_train = pd.read_csv(path_train_03,dtype={'custno':str})
# print(df_train)
df_this_three_month = pd.read_csv(path_this_month_0331,dtype={'custno':str,'fundcode':str})
# print(df_this_three_month)
df_next_month = pd.read_csv(path_goal_0331_next_month,dtype={'custno':str,'fundcode':str})
df_next_month.rename(columns={'counts':'下个月浏览的次数','amount':'下个月购买的金额'},inplace=True)
# print(df_next_month)
df_fund_score = pd.read_csv(path_fund_score_stock,dtype={'custno':str,'fundcode':str})
df_fund_score.fillna(0,inplace=True)
# print(df_fund_score)
df_this_three_month.fillna(0,inplace=True)
# print("**************************************")
# print(df_this_three_month)

##用户的基础信息
df = pd.merge(df_train,df_this_three_month,on='custno',how='outer')
# print(df)

#合并两者
#用户基础数据和交易浏览数据和用户对基金的评分合并
df =pd.merge(df,df_fund_score,on=['custno','fundcode'],how='left')
# print(df)

df_next_month = df_next_month[df_next_month['下个月浏览的次数'] > 0]

#目标函数合并
df= pd.merge(df,df_next_month,on=['custno','fundcode'],how='outer')
# print(df)

df['下个月购买的金额'] = [1 if x > 0 else 0 for x in df['下个月购买的金额']]
df['下个月浏览的次数'] = [1 if x > 0 else 0 for x in df['下个月浏览的次数']]
# print(df)
df.fillna(0,inplace=True)
print(df)
ycol_buy = '下个月购买的金额'
ycol_browse = '下个月浏览的次数'
idx = 'custno'
idfund = 'fundcode'

feature_name = list(filter(lambda x: x not in [idx, ycol_buy,ycol_browse,idfund], df.columns))

import lightgbm as lgb
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},

    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
num_leaf = 64

# lgb_model = lgb.train(lgb_params,lgbtrain,valid_sets=lgbvalid,evals_result=evals_result,num_boost_round=1000,early_stopping_rounds=60,verbose_eval=50,categorical_feature=categorical_feature_lists)
#create dataset for lgb
click = df['下个月浏览的次数']

#CV
ycol_buy = '下个月购买的金额'
ycol_browse = '下个月浏览的次数'
kfolder = KFold(n_splits=5, shuffle=True, random_state=2019)
for fold_id, (trn_idx, val_idx) in enumerate(kfolder.split(df)):
    print(f'\nFold_{fold_id} Training ================================\n')
    X_train=df.iloc[trn_idx][feature_name]
    y_train = df.iloc[trn_idx][ycol_browse]
    X_test = df.iloc[val_idx][feature_name]
    y_test = df.iloc[val_idx][ycol_browse]
    lgb_train = lgb.Dataset(X_train,y_train)
    lgb_eval = lgb.Dataset(X_test,y_test,reference=lgb_train)
    #开始训练
    print("Start training.....")
    gbm = lgb.train(params,lgb_train,num_boost_round=1000,valid_sets=lgb_train)
    y_pred_train = gbm.predict(X_train,pred_leaf=True)
    #One-hot编号
##===================== 训练集转换
    print('Writing transformed training data')
    transformed_training_matrix = np.zeros([len(y_pred_train), len(y_pred_train[0]) * num_leaf],dtype=np.int64)  # N * num_tress * num_leafs
    for i in range(0, len(y_pred_train)):
        temp = np.arange(len(y_pred_train[0])) * num_leaf + np.array(y_pred_train[i])
        transformed_training_matrix[i][temp] += 1
    print("************")
    print(transformed_training_matrix.shape)
##===================== 测试集转换
    print('Writing transformed testing data')
    y_pred_test = gbm.predict(X_test, pred_leaf=True)
    transformed_testing_matrix = np.zeros([len(y_pred_test), len(y_pred_test[0]) * num_leaf], dtype=np.int64)
    for i in range(0, len(y_pred_test)):
        temp = np.arange(len(y_pred_test[0])) * num_leaf + np.array(y_pred_test[i])
        transformed_testing_matrix[i][temp] += 1
# LR

    print("Logistic Regression Start")
    c = np.array([1])
    for t in range(0, len(c)):
        print('start lr')
        lm = LogisticRegression(penalty='l2', C=c[t],n_jobs=-1)
        lm.fit(transformed_training_matrix, y_train)
        # y_pred_label = lm.predict(transformed_training_matrix )
        y_pred_label = lm.predict(transformed_testing_matrix)
        y_pred_est = lm.predict_proba(transformed_testing_matrix)
        print('start compute loss')
        ##计算准确率
        train_pred = pd.DataFrame({
            'true': df[ycol_browse],
            'pred': np.zeros(len(df))})
        #设置阈值来处理
        threshold = 0.5
        y_pred_label = [1 if pred > threshold else 0 for pred in y_pred_label]

        train_pred.loc[val_idx, 'pred'] = y_pred_label

        from sklearn.metrics import accuracy_score
        from sklearn import metrics
        from sklearn.metrics import roc_auc_score
        # from sklearn.metrics import accuracy_score
        print('AUC: %.4f' % metrics.roc_auc_score(train_pred['true'], train_pred['pred']))
        print('ACC: %.4f' % metrics.accuracy_score(train_pred['true'], train_pred['pred']))
        print('Recall: %.4f' % metrics.recall_score(train_pred['true'], train_pred['pred']))
        print('F1-score: %.4f' % metrics.f1_score(train_pred['true'], train_pred['pred']))
        print('Precesion: %.4f' % metrics.precision_score(train_pred['true'], train_pred['pred']))
        print(metrics.confusion_matrix(train_pred['true'], train_pred['pred']))