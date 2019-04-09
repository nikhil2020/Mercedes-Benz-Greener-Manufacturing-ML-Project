#!/usr/bin/env python
# coding: utf-8

# # Mercedes-Benz Greener Manufacturing

# Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include the passenger safety cell with the crumple zone, the airbag, and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium carmakers. Mercedes-Benz cars are leaders in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams.
# 
# 

# # Importing the libraries

# In[43]:


import time
import random
from math import *
import operator
import pandas as pd
import numpy as np

# import plotting libraries
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set(font_scale=1.5)

# import the ML algorithm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
#from pandas.core import datetools

# import libraries for model validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# import libraries for metrics and reporting
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings('ignore')


# In[44]:


#import os 
#os.getcwd()
# os.chdir('/Users/ds/Desktop/Project/Mercedes_Benz_Greener_Manufacturing')


# In[45]:


pwd


# ### Importing the Titanic dataset

# In[46]:


df = pd.read_csv('train.csv')


# ## Data understanding and Exploration

# In[47]:


df.head()


# In[48]:


df.shape


# In[49]:


df.columns


# In[50]:


df.describe()


# ### Checking for the variance of all features and removing the feature which is equal to zero

# In[51]:


df.var()==0


# ### Droping the feature whose variance is equal to zero

# In[52]:


df.drop(['X11'],axis =1,inplace =True)


# ### Checking for null values 

# In[53]:


df.isnull().sum()


# ### count of the unique values in each feature

# In[54]:


df.nunique().head(10)


# ### info gives the information of the data types, no. of columns,no. of rows

# In[55]:


df.info()


# In[56]:


df.dtypes


# In[57]:


df.shape


# In[58]:


y = df.iloc[:,1].values


# In[59]:


X = df.iloc[:,2:377].values


# In[60]:


X.shape


# # Label Enncoder

# ### Converting the nomial features object data types into numerical 

# In[61]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder = LabelEncoder()
X[:,0] = encoder.fit_transform(X[:,0])
X[:,1] = encoder.fit_transform(X[:,1])
X[:,2] = encoder.fit_transform(X[:,2])
X[:,3] = encoder.fit_transform(X[:,3])
X[:,4] = encoder.fit_transform(X[:,4])
X[:,5] = encoder.fit_transform(X[:,5])
X[:,6] = encoder.fit_transform(X[:,6])
X[:,7] = encoder.fit_transform(X[:,7])


# In[62]:


X


# In[63]:


#X = pd.DataFrame(X)


# In[64]:


X.shape


# # PCA dimensionality reduction

# In[65]:


from sklearn.decomposition import PCA


# In[66]:


X_centered = X - X.mean(axis=0)


# In[67]:


pca = PCA(n_components=3)
pca.fit(X_centered)


# In[68]:


X_pca = pca.transform(X_centered)


# In[69]:


X_pca.shape


# In[70]:


pca.components_


# In[71]:


pca.explained_variance_


# In[72]:


pca.explained_variance_ratio_


# # Splitting the dataset into traning and testing 

# ### 70% of the dataset is goes for training

# ### 30% of the dataset is goes for testing

# In[73]:


get_ipython().run_line_magic('time', '')
# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_pca,y,test_size =0.3,random_state=1)


# In[74]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Random Forest Regressor

# In[79]:


get_ipython().run_line_magic('time', '')
# Importing the RF model from scikit learn ensemble 
from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor()
model = classifier.fit(X_train,y_train)
model


# In[80]:


# Predicting the values on independent variables testing dataset
y_pred = classifier.predict(X_test)


# In[81]:


# Model evaluation metrics for regression

print('Mean Abs Error   MAE    : ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Sq  Error MSE      : ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Sq Error RMSE : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r2 value                : ', metrics.r2_score(y_test, y_pred))


# # XGBoost Model Regressor

# In[82]:


get_ipython().run_line_magic('time', '')
# Importing the XGBoost model from scikit learn ensemble 
from xgboost import XGBRegressor
classifier = XGBRegressor(n_estimator =1000)
classifier.fit(X_train,y_train)


# In[83]:


# Predicting the values on independent variables testing dataset
y_pred = classifier.predict(X_test)


# In[84]:


# Model evaluation metrics for regression

print('Mean Abs Error   MAE    : ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Sq  Error MSE      : ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Sq Error RMSE : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r2 value                : ', metrics.r2_score(y_test, y_pred))


# # The Champion model out of all Models is XGBoost
