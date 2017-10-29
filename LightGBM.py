
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import lightgbm as lgb
import random


# In[ ]:

train = pd.read_csv("./train_ffm", header = None)
test1 = pd.read_csv("./test_ffm_wk1", header = None)
test2 = pd.read_csv("./test_ffm_wk2", header = None)
print train.shape, test1.shape, test2.shape


# In[ ]:

random.seed(2)


# In[ ]:

dev_cm = set(random.sample(list(set(train[0])), int(0.8*train[0].nunique())))
val_cm = set(train[0]) - dev_cm
print train[0].nunique(), len(dev_cm), len(val_cm)


# In[ ]:

dev_x, dev_y = train[train[0].isin(dev_cm)].ix[:, 4:], train[train[0].isin(dev_cm)][3]
val_x, val_y = train[train[0].isin(val_cm)].ix[:, 4:], train[train[0].isin(val_cm)][3]
train_x, train_y = train.ix[:, 4:], train[3]

dev_group =  train[train[0].isin(dev_cm)][[0,1,2]].groupby([0,2]).count().reset_index()[1].values
val_group =  train[train[0].isin(val_cm)][[0,1,2]].groupby([0,2]).count().reset_index()[1].values
train_group = train[[0,1,2]].groupby([0,2]).count().reset_index()[1].values


# In[ ]:

train[train[0].isin(dev_cm)][[0,1,2]].head(20)


# In[ ]:

lgb_dev = lgb.Dataset(train[train[0].isin(dev_cm)].ix[:, 112:], train[train[0].isin(dev_cm)][3])
lgb_val = lgb.Dataset(train[train[0].isin(val_cm)].ix[:, 112:], train[train[0].isin(val_cm)][3])
lgb_train = lgb.Dataset(train.ix[:, 112:], train[3])


# In[ ]:

params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'binary', 
          'nthread': 16, 
          'verbose': 1,
          'num_leaves': 300, 
          'learning_rate': 0.02, 
          'max_bin': 32, 
          'subsample_for_bin': 200000,
          'subsample': 0.8, 
          'subsample_freq': 2, 
          'colsample_bytree': 0.8, 
          'reg_alpha': 0.00005, 
          'reg_lambda': 0.00001,
          'min_split_gain': 0.0, 
          'min_child_weight': 1, 
          'min_child_samples': 5, 
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_logloss'
         }

model = lgb.train(params, lgb_dev, num_boost_round=1000, valid_sets = (lgb_val), verbose_eval = True,
                  early_stopping_rounds = 20)


# In[ ]:

features = pd.read_csv("/axp/rim/imml/warehouse/ujjwalsrao/orchestra/benchmark_oet/Insample_159offers/6_model_ffm/1_output/feature_mapping.csv")

sorted([(f,s) for f,s in zip(features['feature'], model.feature_importance())], key = lambda x : x[1], reverse = True)


# In[ ]:

model = lgb.train(params, lgb_train, num_boost_round=342)
pred1 = model.predict(test1.ix[:,4:])
pred2 = model.predict(test2.ix[:,4:])


# In[ ]:

test1['pred'] = pred1
test2['pred'] = pred2
test = test1.append(test2)


# In[ ]:

scores_final = test.pivot_table(index=[0,2],columns=[1],values='pred',fill_value=-1).reset_index()


# In[ ]:

scores_final.head()


# In[ ]:

LOOKUP = '/axp/rim/imml/warehouse/ujjwalsrao/orchestra/benchmark_oet/Insample_159offers/1_data/'
offer_list = pd.read_csv(LOOKUP + 'offer_lookup', header=None, sep='\t')
offer_list.columns = ['offer_id','offer_content_id']
offer_list.head()


# In[ ]:

scores_final = scores_final[[0,2] + list(offer_list['offer_content_id'])]
scores_final.head()


# In[ ]:

scores_final.to_csv("./lgb_scores.csv", header = None, index = False)


# In[ ]:

list(features.feature)[112:]


# In[ ]:

model = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1,
    n_estimators=10, max_bin=255, subsample_for_bin=50000,
    objective=None, min_split_gain=0, min_child_weight=5,
    min_child_samples=10, subsample=1, subsample_freq=1, colsample_bytree=1,
    reg_alpha=0, reg_lambda=0, seed=0, nthread=-1,
    silent=True)

model.fit()


# In[ ]:

model = lgb.LGBMRanker()


# In[ ]:



