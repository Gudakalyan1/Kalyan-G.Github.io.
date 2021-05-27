#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.stats import skew


# In[12]:


df=pd.read_csv("Downloads\linear regression machine learning.csv")


# In[17]:


df


# In[30]:


#plotting scatter plot to check wheather the relationship exist or not
plt.figure(figsize=(20,10))
sns.regplot(x='year',y='income',data=df)
plt.xlabel('years')
plt.ylabel('income per capita in usd')
plt.show()


# In[15]:


df.corr()


# In[29]:


#checking for skewness
sns.distplot(df['income'],hist=False)
            
plt.show()
print(skew(df['income']))


# In[39]:


reg=linear_model.LinearRegression()
reg.fit(df[['year']],df.income)


# In[43]:


reg.predict([[2017]])


# In[45]:


income=pd.read_csv("Downloads\income.csv")


# In[46]:


income


# In[49]:


prediction=reg.predict(income)
predictions


# In[52]:


#saving to excel
income['income']=predictions
income.to_csv("income.csv",index=False)


# In[ ]:




