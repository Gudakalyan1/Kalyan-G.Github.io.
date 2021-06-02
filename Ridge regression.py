#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


# In[34]:


df=pd.read_csv("Downloads\melborne_housing data.csv")


# In[35]:


df.head()


# In[36]:


df.dtypes


# In[37]:


df.isna().sum()


# In[38]:


melborn=df.copy()


# In[39]:


melborn.drop(['Address','Date','Postcode','YearBuilt','Lattitude','Longtitude','Regionname'],axis=1,inplace=True)


# In[40]:


melborn.head()


# In[41]:


melborn.isna().sum()


# In[42]:


melborn['BuildingArea'].fillna(melborn.BuildingArea.mean(),axis=0,inplace=True)
melborn['CouncilArea'].fillna(melborn.CouncilArea.mode(),axis=0,inplace=True)
melborn['Car'].fillna(melborn.Car.fillna(0),axis=0,inplace=True)


# In[43]:


melborn[melborn["CouncilArea"]=='Unavailable']


# In[44]:


melborn.CouncilArea.unique()


# In[45]:


melborn['CouncilArea'].isna().sum()


# In[46]:


melborn.dropna(inplace=True)


# In[47]:


melborn['CouncilArea'].str.strip()


# In[48]:


melborn['Suburb'].str.strip()


# In[49]:


melborn.head()


# In[50]:


melborn.corr()


# In[51]:


model=pd.get_dummies(melborn,drop_first="True")


# In[52]:


melborn


# In[53]:


x=model.drop(['Price'],axis=1)
y=model['Price']
x


# In[54]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)


# In[55]:


reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)


# In[56]:


reg.score(x_train,y_train)


# In[57]:


reg.score(x_test,y_test)


# In[67]:


cross_val_score(ridge_reg,x,y,cv=3)


# In[58]:


lasso_reg=linear_model.Lasso(alpha=50,max_iter=100,tol=0.2)
lasso_reg.fit(x_train,y_train)


# In[59]:


lasso_reg.score(x_train,y_train)


# In[60]:


lasso_reg.score(x_test,y_test)


# In[61]:


ridge_reg=linear_model.Ridge(alpha=50,max_iter=100,tol=0.2)
ridge_reg.fit(x_train,y_train)


# In[62]:


ridge_reg.score(x_train,y_train)


# In[63]:


ridge_reg.score(x_test,y_test)


# In[68]:


model


# In[79]:


p=ridge_reg.predict(x_train)


# In[ ]:





# In[ ]:




