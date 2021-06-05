#!/usr/bin/env python
# coding: utf-8

# In[67]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[3]:


df=pd.read_csv("Downloads\house_sales us.csv")


# In[4]:


df.head()


# In[6]:


#checking for null values 
df.isna().sum()


# In[8]:


#data types
df.dtypes


# In[14]:


df.drop(['id'],axis=1,inplace=True)
df.describe()


# In[12]:


df.floors.value_counts()


# In[15]:


df.corr()


# In[25]:


plt.figure(figsize=(10,5))
sns.boxplot(x='waterfront',y='price',data=df)


plt.show()


# In[27]:


plt.figure(figsize=(10,5))

sns.regplot(x="sqft_above",y="price",data=df)
plt.show()


# In[43]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[74]:


x=df[["floors",
"waterfront",
"lat",
"bedrooms",
"sqft_basement",
"view",
"bathrooms",
"sqft_living15",
"sqft_above",
"grade",
"sqft_living"]]
y=df['price']
#fitting the model
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
yhat=reg.predict(x_train)


# In[58]:


train=reg.score(x_train,y_train)
test=reg.score(x_test,y_test)
train


# In[59]:


test


# In[79]:


Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',linear_model.LinearRegression())]


# In[80]:


pipe=Pipeline(input)
pipe


# In[82]:


pipe.fit(x,y)


# In[83]:


pipe.score(x,y)


# In[84]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[86]:


ridge=linear_model.Ridge(alpha=1)
ridge.fit(x_train,y_train)


# In[87]:


ridge.score(x_train,y_train)


# In[88]:


ridge.score(x_test,y_test)


# In[92]:


poly=PolynomialFeatures(degree=2)
x_train_pr=poly.fit_transform(x_train)
x_test_pr=poly.fit_transform(x_test)


# In[107]:


ridge1=linear_model.Ridge(alpha=0.1)
ridge1.fit(x_train_pr,y_train)


# In[109]:


y_hat=ridge1.predict(x_train_pr)
print('true_values:',y_test.values)
print('predicted_values:',y_hat)


# In[110]:


ridge1.score(x_test_pr,y_test)


# In[ ]:





# In[ ]:




