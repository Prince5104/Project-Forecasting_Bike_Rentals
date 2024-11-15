#!/usr/bin/env python
# coding: utf-8

# **End to End Project - Bikes Assessment - Basic - Importing the libraries**

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import matplotlib.pyplot as plt
import os


# In[2]:


np.random.seed(42)


# **Loading the data**

# In[3]:


filePath = ("/cxldata/datasets/project/bikes.csv")


# In[4]:


bikesData = pd.read_csv(filePath)


# In[5]:


bikesData.head()


# **Perform EDA on the Dataset**

# In[6]:


bikesData.info()


# In[7]:


bikesData['yr'].value_counts()


# In[8]:


bikesData.describe()


# **Cleaning the data - Dropping unwanted features**

# In[9]:


columnsToDrop = ['instant', 'casual', 'registered', 'atemp', 'dteday']


# In[10]:


bikesData = bikesData.drop(columnsToDrop, axis = 1)


# In[11]:


bikesData.head()


# **Divide Dataset into Train and Test set**

# In[12]:


np.random.seed(42)


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


bikesData['dayCount'] = pd.Series(range(bikesData.shape[0]))/24


# In[15]:


bikesData.head()


# In[16]:


train_set, test_set = train_test_split(bikesData, test_size=0.3, random_state=42)


# In[17]:


import warnings
warnings.filterwarnings(action= 'ignore')


# In[18]:


train_set.sort_values('dayCount', axis= 0, inplace=True)
test_set.sort_values('dayCount', axis= 0, inplace=True)


# In[19]:


train_set.head()


# In[20]:


train_set.shape


# In[21]:


test_set.shape


# In[22]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# **Feature Scale the Dataset**   (Basic - Cleaning the data - Feature Scaling)

# In[23]:


columnsToScale = ['temp', 'hum', 'windspeed']


# In[24]:


scaler = StandardScaler()


# In[25]:


train_set[columnsToScale] = scaler.fit_transform(train_set[columnsToScale])
test_set[columnsToScale] = scaler.transform(test_set[columnsToScale])


# In[27]:


display_scores(train_set)
display_scores(test_set)


# **Train various Models on the Dataset**

# In[29]:


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[32]:


trainingCols = train_set.drop('cnt', axis=1)
trainingCols.head()


# In[34]:


trainingLabels = train_set['cnt'].copy()
trainingLabels.head()


# **Train DecisionTree Model**

# In[45]:


dec_reg = DecisionTreeRegressor(random_state=42)


# In[47]:


dt_mae_scores = -cross_val_score(dec_reg,
                               trainingCols,trainingLabels,
                               cv=10, scoring="neg_mean_absolute_error")
display_scores(dt_mae_scores)


# In[53]:


dt_mse_scores = np.sqrt(-cross_val_score(dec_reg,
                               trainingCols,trainingLabels,
                               cv=10, scoring="neg_mean_squared_error"))
display_scores(dt_mse_scores)


# **Train Linear Regression Model**

# In[58]:


lin_reg = LinearRegression()


# In[59]:


lr_mae_scores = -cross_val_score(lin_reg, trainingCols, trainingLabels,
                                cv=10, scoring="neg_mean_absolute_error")
display_scores(lr_mae_scores)


# In[60]:


lr_mse_scores = np.sqrt(-cross_val_score(lin_reg, trainingCols, trainingLabels,
                                cv=10, scoring="neg_mean_squared_error"))
display_scores(lr_mse_scores)


# **Train Random Forest Model**

# In[65]:


forest_reg = RandomForestRegressor(n_estimators=150, random_state=42)


# In[66]:


rf_mae_scores = -cross_val_score(forest_reg, trainingCols, trainingLabels,
                                cv=10, scoring="neg_mean_absolute_error")
display_scores(rf_mae_scores)


# In[67]:


rf_mse_scores = np.sqrt(-cross_val_score(forest_reg, trainingCols, trainingLabels,
                                cv=10, scoring="neg_mean_squared_error"))
display_scores(rf_mse_scores)


# **Fine Tune the Models**  (Choosing set of hyperparameter combinations for Grid Search)

# In[72]:


from sklearn.model_selection import GridSearchCV


# In[79]:


param_grid = [{'n_estimators': [120, 150], 'max_features':[10,12], 'max_depth':[15, 28]}]


# **Defining GridSearchCV**

# In[85]:


grid_search = GridSearchCV(forest_reg, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error")


# **Run GridSearchCV**

# In[88]:


grid_search.fit(trainingCols, trainingLabels)


# In[90]:


print("Best hyperparameters", grid_search.best_params_)
print("Best Estimator", grid_search.best_estimator_)
print("Best Score (Negative MSE):", grid_search.best_score_)


# **Knowing Feature Importances**

# In[95]:


feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)


# **Evaluate the Models** (Preparing to test the final model on Test dataset)

# In[99]:


final_model = grid_search.best_estimator_


# In[100]:


test_set.sort_values('dayCount', axis= 0, inplace=True)


# In[103]:


test_x_cols = (test_set.drop('cnt', axis=1)).columns.values


# In[107]:


test_x_cols


# In[114]:


test_y_cols = 'cnt'


# In[116]:


X_test = test_set.loc[:,test_x_cols]


# In[119]:


y_test = test_set.loc[:, test_y_cols]


# **Make Predictions on the Test dataset using Final Model**

# In[129]:


test_set.loc[:, 'predictedCounts_test'] = final_model.predict(X_test)


# In[131]:


mse = mean_squared_error(y_test, test_set.loc[:,'predictedCounts_test'])


# In[134]:


final_mse = np.sqrt(mse)
final_mse


# In[135]:


test_set.head()


# In[136]:


times = [9,18]
for time in times:
    fig = plt.figure(figsize=(8, 6))
    fig.clf()
    ax = fig.gca()
    test_set_freg_time = test_set[test_set.hr == time]
    test_set_freg_time.plot(kind = 'line', x = 'dayCount', y = 'cnt', ax = ax)
    test_set_freg_time.plot(kind = 'line', x = 'dayCount', y = 'predictedCounts_test', ax =ax)
    plt.show()


# In[ ]:





# In[ ]:




