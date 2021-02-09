import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, GridSearchCV
from matplotlib import pyplot as plt
from features.features import Feature

model_folder = '../data/model/LGB/'


def get_group(fold_part):
    fold_part['group'] = fold_part.groupby('query_id')['query_id'].transform('size')
    group = fold_part[['query_id', 'group']].drop_duplicates()['group']
    return group


def get_eval_group(fold_part):
    group = fold_part.groupby('query_id')['query_id'].transform('size')
    return group


train_data = pd.read_csv('../data/train.csv', encoding='utf-8')
feature = Feature(train_data)
training_pd = feature.get_feature()
feat_name = []
for col in training_pd:
    if col != 'relevence':
        feat_name.append(col)

for feat in feat_name[4:]:
    training_pd[feat] = pd.to_numeric(training_pd[feat])
    # validation_pd[feat] = pd.to_numeric(validation_pd[feat])

# Partial data frames used for quicker gorupby
training_pd_part = training_pd[['query_id', 'doc_id']]


def get_dataset(df):
    return df[feat_name[4:]], df['relevence']


train_x, train_y = get_dataset(training_pd)
train_group = get_group(training_pd_part)
# valid_x, valid_y = get_dataset(validation_pd)
# valid_group = get_group(validation_pd_part)
# valid_group_full = get_eval_group(validation_pd_part)

params = {
    'n_estimators': 10,
    'boosting_type': 'gbdt',
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_at': [10],
    # 'lambda_l1': 1,
    # 'lambda_l2': 0.1,
    'learning_rate': 0.05,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.8,
    # 'bagging_freq': 10,
    # 'tree_learner': 'feature',
    # 'verbose_eval': 200,
    'metric_freq': 10,
    'train_metric': True,
    'num_leaves': 120,
    'max_depth': 7,
    # 'min_child_samples': 30,
    'random_state': 2020,
    'num_threads': 16,
    # 'early_stopping_rounds': 50,
    'max_bin': 5,
    'min_data_in_left': 75,
}

'''
    Train LightGBM model
'''


# def lgb_train(hyper_params, train_x, train_y, valid_x, valid_y, group_train, group_valid):
#     training_set = lgb.Dataset(train_x, label=train_y, group=group_train)
#     validation_set = lgb.Dataset(valid_x, valid_y, reference=training_set, group=group_valid)
#     model = lgb.train(hyper_params, training_set, valid_sets=[training_set, validation_set], verbose_eval=False)
#     return model

def lgb_train(hyper_params, train_x, train_y, group_train):
    training_set = lgb.Dataset(train_x, label=train_y, group=group_train)
    # validation_set = lgb.Dataset(valid_x, valid_y, reference=training_set, group=group_valid)
    # model = lgb.train(hyper_params, training_set, valid_sets=[training_set, validation_set], verbose_eval=False)
    model = lgb.train(hyper_params, training_set, verbose_eval=False)
    return model


'''
    Cross-validation implementation based on sklearn
    * 5 Folds
'''
# def lgb_cv_skl(train_x, train_y, test_x, n_splits=5):
#     folds = KFold(n_splits=n_splits, shuffle=False)
#     oof = np.zeros(train_x.shape[0])
#     test_preds = np.zeros(test_x.shape[0])
#     for fold_num, (train_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
#         print("Fold Number: {}".format(fold_num + 1))
#         # Get training fold and validation fold
#         train_fold_x, train_fold_y = train_x.iloc[train_idx], train_y.iloc[train_idx]
#         validation_fold_x, validation_fold_y = train_x.iloc[val_idx], train_y.iloc[val_idx]
#         # Get training groups
#         train_group = get_group(training_pd_part.iloc[train_idx])
#         validation_group = get_group(training_pd_part.iloc[val_idx])
#         # Train model in this fold
#         model = lgb_train(params, train_fold_x, train_fold_y, validation_fold_x, validation_fold_y, train_group, validation_group)
#         # Save model
#         model_path = f'{model_folder}fold_{fold_num}_model'
#         pd.to_pickle(model, model_path)
#         pickled_model = pd.read_pickle(model_path)
#         oof[val_idx] = pickled_model.predict(validation_fold_x)
#         test_preds += pickled_model.predict(test_x)
#     return oof, test_preds / n_splits

#
# def MRR(indices_k, target, k=10):
#     """
#     Compute mean reciprocal rank.
#     :param logits: 2d array [batch_size x rel_docs_per_query]
#     :param target: 2d array [batch_size x rel_docs_per_query]
#     :return: mean reciprocal rank [a float value]
#     """
#     assert indices_k.shape == target.shape
#     # num_doc = logits.shape[1]
#     # indices_k = np.argsort(-logits, 1)[:, :k]  # 取topK 的index   [n, k]
#
#     reciprocal_rank = 0
#     for i in range(target.shape[0]):
#         for j in range(target.shape[1]):
#             idx = np.where(indices_k[i] == target[i][j])[0]
#             if len(idx) != 0:
#                 assert len(idx) == 1
#                 reciprocal_rank += 1.0 / (idx[0] + 1)
#                 break
#     return reciprocal_rank / indices_k.shape[0]
#
# def get_np_pred(metric='pred'):
#     test_labels = np.zeros((len(label_pd), 10))
#     idx = 0
#     for name, group in test_pd_group:
#         group = group.sort_values(metric, ascending=False).head(10)
#         query_label = group['doc_id'].values.tolist()
#         test_labels[idx] = np.array(query_label)
#         idx += 1
#     return test_labels
#
# def get_pred_mrr(model):
#     test_pd['pred'] = model.predict(feature_pd[feat_name])
#     mrr = MRR(get_np_pred('pred'), eval_labels)
#     return mrr

# Single LGB

model = lgb_train(params, train_x, train_y, train_group)
# single_model = lgb_train(params, train_x, train_y, valid_x, valid_y, train_group, valid_group)
# get_pred_mrr(single_model)
model_path = model_folder + 'single_model'
pd.to_pickle(model, model_path)

model.save_model("../data/lightgbm.txt")
model = pd.read_pickle(model_path)

# trim_name = [trim[5:] for trim in feat_name]
# trim_name =[]
# importances = model.feature_importance()
#
# plt.figure(figsize=(12,6))
# plt.ylabel('Importance')
# plt.title('LightGBM Feature Importances')
# plt.bar(x = range(len(trim_name)),
#         height = importances,
#         tick_label = trim_name,
#         color = 'dodgerblue',
#         width = 0.8
#        )
# plt.xticks(rotation=90)
# for x,y in enumerate(importances):
#     plt.text(x,y+0.1,'%s' %round(y,1),ha='center')
#
# plt.title("Feature Importances")
# plt.savefig('./lgb_feature.png')
# plt.show()


# Cross-validtion
# cv_ret = lgb_cv_skl(train_x, train_y, valid_x, n_splits=5)
