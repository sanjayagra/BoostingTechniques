
# coding: utf-8

# In[68]:

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# ## Reading the data

# In[69]:

path = '/Users/sanjayagrawal/Downloads/Kaggle/zillow/'
sample_submission = pd.read_csv(path + 'sample_submission.csv')
train_2016_v2 = pd.read_csv(path + 'train_2016_v2.csv')
properties_2016 = pd.read_csv(path + 'properties_2016.csv')


# In[70]:

for i in properties_2016.columns:
    if properties_2016.dtypes[i] == 'object':
        print (i)
        properties_2016[i] = properties_2016[i].map(lambda x : str(x))
        le = preprocessing.LabelEncoder()
        properties_2016[i] = le.fit_transform(properties_2016[i])


# In[71]:

sample_submission = pd.melt(sample_submission, id_vars=['ParcelId'], value_vars=['201610', '201611', '201612', '201710', '201711', '201712'])
sample_submission = sample_submission.drop('value', axis=1)


# In[103]:

train_2016_v2 = train_2016_v2.rename(columns = {'parcelid':'ParcelId'})
properties_2016 = properties_2016.rename(columns = {'parcelid':'ParcelId'})
train = train_2016_v2.merge(properties_2016, on='ParcelId', how = 'inner')
test = sample_submission.merge(properties_2016, on='ParcelId', how = 'inner')
train = train.fillna(-999)
test = test.fillna(-999)


# ## Feature Engineering

# In[104]:

import datetime
train['month'] = train['transactiondate'].map(lambda x : (datetime.datetime.strptime(x, "%Y-%m-%d")).month)
test['month'] = test['variable'].map(lambda x : int(x[4:6]))


# In[105]:

x = train.drop(['ParcelId', 'logerror', 'transactiondate'], axis=1)
y= train['logerror']

x_train, y_train, x_vald, y_vald = train_test_split(x, y, train_size=0.80, random_state=1234)

x_test = test.drop(['ParcelId', 'variable'], axis=1)
print (x.shape, x_test.shape)


# ## LightGBM Model

# In[90]:

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 40,
    'learning_rate': 0.03,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.6,
    'bagging_freq': 2,
    'max_depth' : -1,
    'max_bin' : 32,
    'verbose': 0
}

# lgb_train = lgb.Dataset(x_train, x_vald)
# lgb_eval = lgb.Dataset(y_train, y_vald, reference=lgb_train)
# gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_eval, early_stopping_rounds=100)


# In[108]:

estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.03],
    'n_estimators': [40],
    'feature_fraction': [0.9],
    'task': ['train'],
    'num_leaves' : [40],
    'boosting_type': ['gbdt'],
    'objective': ['regression'],
    'metric': [{'l2', 'auc'}],
    'bagging_fraction': [0.6],
    'bagging_freq': [2],
    'max_depth' : [-1],
    'max_bin' : [32],
    'verbose': [0]
}

gbm = GridSearchCV(estimator, param_grid)
gbm.fit(x, y)
print('Best parameters found by grid search are:', gbm.best_params_)


# In[107]:

# gbm.feature_importance()
# k=0
# list = []
# for i in x.columns:
#     if gbm.feature_importance()[k]==0:
#         print (i, gbm.feature_importance()[k])
#         list.append(i)
#     k=k+1

# print (list)
x = x.drop(list, axis=1)
x_test = x_test.drop(list, axis=1)
print (x.shape, x_test.shape)


# ## Predictions

# In[109]:

# gbm.save_model('model.txt')
y_pred = gbm.predict(x_test)
Predicted = test
Predicted['pred'] = y_pred


# In[110]:

Predicted = Predicted[['ParcelId', 'variable', 'pred']]
Predicted.head()


# In[111]:

Predicted1 = Predicted.pivot(index='ParcelId', columns='variable', values='pred').reset_index()
path = '/Users/sanjayagrawal/Downloads/Kaggle/zillow/result/'
Predicted1.to_csv(path + 'final12_month.csv', index=False)


# In[112]:

Predicted1.head()

