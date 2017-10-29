import xgboost as xgb

import pandas as pd 

import numpy as np

from sklearn.metrics import roc_curve, auc



INPATH  = '/axp/rim/imml/warehouse/ujjwalsrao/orchestra/benchmark_oet/4_model_data/1_output/'

OUTPATH = '/axp/rim/imml/warehouse/ujjwalsrao/orchestra/benchmark_oet/5_model_xgb/1_output/'



train_data = pd.read_csv(INPATH + 'train_xgb')

train_data = train_data.sort_values(by=['cust_xref_id','event_ts']).reset_index(drop=True)

train_groups = train_data[['cust_xref_id','event_ts','click']].groupby(['cust_xref_id','event_ts']).count().reset_index()

train_groups = train_groups['click'].values

train_matrix = xgb.DMatrix(train_data.ix[:,4:], label=train_data['click'], feature_names=train_data.columns[4:])

train_matrix.set_group(train_groups)



valid_data = pd.read_csv(INPATH + 'valid_xgb')

valid_data = valid_data.sort_values(by=['cust_xref_id','event_ts']).reset_index(drop=True)

valid_groups = valid_data[['cust_xref_id','event_ts','click']].groupby(['cust_xref_id','event_ts']).count().reset_index()

valid_groups = valid_groups['click'].values

valid_matrix = xgb.DMatrix(valid_data.ix[:,4:], label=valid_data['click'], feature_names=valid_data.columns[4:])

valid_matrix.set_group(valid_groups)



test_data = pd.read_csv(INPATH + 'test_xgb')

test_data = test_data.sort_values(by=['cust_xref_id','event_ts']).reset_index(drop=True)

test_groups = test_data[['cust_xref_id','event_ts','click']].groupby(['cust_xref_id','event_ts']).count().reset_index()

test_groups = test_groups['click'].values

test_matrix = xgb.DMatrix(test_data.ix[:,4:], label=test_data['click'], feature_names=test_data.columns[4:])

test_matrix.set_group(test_groups)



print 'data read...'

print train_data.shape, valid_data.shape, test_data.shape



# XGBoost Model

param = {}

param['booster'] = 'gbtree'

param['objective'] = 'rank:pairwise'

param['eta'] = 0.10

param['seed']=  1008

param['max_depth'] = 3 

param['min_child_weight'] = 1000

param['silent'] =  1  

param['nthread'] = 12

param['subsample'] = 0.5

param['gamma'] = 1.0

param['colsample_bytree'] = 1.0

param['colsample_bylevel'] = 1.0                  



model = xgb.train(params=param, dtrain=train_matrix, num_boost_round=25)

print 'model fit...'



# Variable Importance

for variable,sore in sorted(model.get_fscore().items(), key=lambda x : x[1], reverse=True)[:10]:

	print variable,sore



# Predictions

prob_train = pd.DataFrame(model.predict(train_matrix), columns=['model_prob'])

prob_valid = pd.DataFrame(model.predict(valid_matrix), columns=['model_prob'])

prob_test  = pd.DataFrame(model.predict(test_matrix), columns=['model_prob'])

print 'model predict...'

print prob_train.shape, prob_valid.shape, prob_test.shape



# Evaluation

fpr,tpr,thresholds = roc_curve(train_data['click'], prob_train)

print 'Train:', 2*auc(fpr,tpr) - 1



fpr,tpr,thresholds = roc_curve(valid_data['click'], prob_valid)

print 'Valid:', 2*auc(fpr,tpr) - 1



fpr,tpr,thresholds = roc_curve(test_data['click'], prob_test)

print 'Test:', 2*auc(fpr,tpr) - 1



# Leaf Preditions

columns = ['bst' + str(x+1) for x in range(25)]

pred_train = pd.DataFrame(model.predict(train_matrix, pred_leaf=True), columns=columns)

pred_valid = pd.DataFrame(model.predict(valid_matrix, pred_leaf=True), columns=columns)

pred_test  = pd.DataFrame(model.predict(test_matrix, pred_leaf=True), columns=columns)

print 'model leaf...'

print pred_train.shape, pred_valid.shape, pred_test.shape



# Join All

pred_train = train_data[['cust_xref_id','offer_content_id','event_ts','click']].join(prob_train, how='inner').join(pred_train, how='inner')

pred_valid = valid_data[['cust_xref_id','offer_content_id','event_ts','click']].join(prob_valid, how='inner').join(pred_valid, how='inner')

pred_test = test_data[['cust_xref_id','offer_content_id','event_ts','click']].join(prob_test, how='inner').join(pred_test, how='inner')

print 'data merged...'

print pred_train.shape, pred_valid.shape, pred_test.shape



# Evaluation

fpr,tpr,thresholds = roc_curve(pred_train['click'], pred_train['model_prob'])

print 'Train:', 2*auc(fpr,tpr) - 1



fpr,tpr,thresholds = roc_curve(pred_valid['click'], pred_valid['model_prob'])

print 'Valid:', 2*auc(fpr,tpr) - 1



fpr,tpr,thresholds = roc_curve(pred_test['click'], pred_test['model_prob'])

print 'Test:', 2*auc(fpr,tpr) - 1





pred_train.to_csv(OUTPATH + 'train_xgb', index = False)

pred_valid.to_csv(OUTPATH + 'valid_xgb', index = False)

pred_test.to_csv(OUTPATH + 'test_xgb', index = False)