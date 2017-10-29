# In[27]:

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor, Pool
import random


# In[28]:

inpath = '/axp/rim/imsadsml/warehouse/sagra39/CatBoost/'
test1 = pd.read_csv(inpath+"valid_xgb_with_leaf_node", header=None)
test2 = pd.read_csv(inpath+"test_xgb_with_leaf_node", header = None)
train = pd.read_csv(inpath+"train_xgb_clicker_with_leaf_node", header = None)
print train.shape, test1.shape, test2.shape


# In[29]:

from sklearn.model_selection import train_test_split
X = train.drop([0,1,2,3], axis=1)
y = train[3]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.80, random_state=1234)

# categorical_features_indices  = [i for i in X.columns if X[i].nunique()<1000]
# print categorical_features_indices 


# In[30]:

# random.seed(2)
# dev_cm = set(random.sample(list(set(train[0])), int(0.8*train[0].nunique())))
# val_cm = set(train[0]) - dev_cm
# print train[0].nunique(), len(dev_cm), len(val_cm)


# In[31]:

# dev_x, dev_y = train[train[0].isin(dev_cm)].ix[:, 4:], train[train[0].isin(dev_cm)][3]
# val_x, val_y = train[train[0].isin(val_cm)].ix[:, 4:], train[train[0].isin(val_cm)][3]
# train_x, train_y = train.ix[:, 4:], train[3]

# dev_group =  train[train[0].isin(dev_cm)][[0,1,2]].groupby([0,2]).count().reset_index()[1].values
# val_group =  train[train[0].isin(val_cm)][[0,1,2]].groupby([0,2]).count().reset_index()[1].values
# train_group = train[[0,1,2]].groupby([0,2]).count().reset_index()[1].values


# In[32]:

# dev_x, dev_y = train[train[0].isin(dev_cm)].ix[:, 4:].values, train[train[0].isin(dev_cm)][3].values
# val_x, val_y = train[train[0].isin(val_cm)].ix[:, 4:].values, train[train[0].isin(val_cm)][3].values
# print "read3"


# In[33]:

# dev_pool = Pool(dev_x, dev_y)
# val_pool = Pool(val_x, val_y)
# test1_pool = Pool(test1.ix[:,4:].values)  
# test2_pool = Pool(test2.ix[:,4:].values)
# print "read4"


# In[34]:

# model = CatBoostClassifier(iterations=6000, 
#                            learning_rate=0.1, 
#                            depth=12,
#                            l2_leaf_reg=9,
#                            gradient_iterations=1,
#                            rsm=0.8,
#                            one_hot_max_size=1,
#                            random_strength=1,
#                            bagging_temperature=1,
#                            thread_count=16,
#                            random_seed=2,
#                            auto_stop_pval=0,
#                            loss_function='CrossEntropy')

model = CatBoostClassifier(iterations=6000, 
                           learning_rate=0.1, 
                           depth=12,
                           l2_leaf_reg=9,
                           gradient_iterations=6,
                           rsm=1,
                           one_hot_max_size=1,
                           random_strength=1,
                           bagging_temperature=1,
                           thread_count=16,
                           random_seed=2,
                           # auto_stop_pval=0,
                           loss_function='CrossEntropy')

# model = CatBoostClassifier(iterations=6000, 
#                            learning_rate=0.1, 
#                            depth=12,
#                            l2_leaf_reg=9,
#                            gradient_iterations=9,
#                            rsm=1,
#                            one_hot_max_size=1,
#                            random_strength=1,
#                            bagging_temperature=1,
#                            thread_count=16,
#                            random_seed=2,
#                            auto_stop_pval=0,
#                            loss_function='CrossEntropy')



# In[ ]:

model.fit(X_train, y_train, use_best_model=True, eval_set=(X_validation, y_validation), verbose=True)
# model.fit(dev_pool, use_best_model=True, eval_set=val_pool, verbose=True)
print 'read5'

model.save_model('/axp/rim/imsadsml/warehouse/sagra39/CatBoost/catboost_xgboost2/model')
# In[21]:

pred1 = model.predict_proba(test1_pool)
pred2 = model.predict_proba(test2_pool)


# In[22]:

test1['pred'] = pred1[:,1]
test2['pred'] = pred2[:,1]
test = test1.append(test2)


# In[23]:

scores_final = test.pivot_table(index=[0,2],columns=[1],values='pred',fill_value=-1).reset_index()


# In[24]:

LOOKUP = '/axp/rim/imsadsml/warehouse/sagra39/FFM/lightGBM/'
offer_list = pd.read_csv(LOOKUP + 'offer_lookup', header=None, sep='\t')
offer_list.columns = ['offer_id','offer_content_id']
offer_list.shape


# In[26]:


# In[25]:

scores_final = scores_final[[0,2] + list(offer_list['offer_content_id'])]
scores_final.to_csv("./catboost_xgboost_new.csv", header = None, index = False)
scores_final.head()
