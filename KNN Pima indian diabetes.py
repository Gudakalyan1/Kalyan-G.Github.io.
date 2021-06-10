#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


#loading the data set
df=pd.read_csv('Downloads\diabetes.csv')


# In[3]:


df.head()


# In[4]:


df.isna().sum()


# In[5]:


df[df.duplicated()]


# In[6]:


df.dtypes


# In[7]:


df.shape


# In[8]:


np.sqrt(768)


# In[5]:


#numpy array for feature and target
x=df.drop('Outcome',axis=1).values
y=df['Outcome'].values


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[7]:


neighbors=np.arange(1,9)
train_accuracy=np.empty(len(neighbors))
test_accuracy=np.empty(len(neighbors))


# In[8]:


for i,k in enumerate(neighbors):
    knn=KNeighborsClassifier(n_neighbors=k)


# In[9]:


knn.fit(x_train,y_train)


# In[10]:


knn.score(x_train,y_train)


# In[11]:


knn.score(x_test,y_test)


# In[13]:


y_hat=knn.predict(x_test)


# In[18]:


cm=metrics.confusion_matrix(y_test,y_hat)


# In[19]:


sns.heatmap(cm)


# In[22]:


df.head()


# In[21]:


knn.predict([[9,136,80,25,89,25.2,0.776,45]])


# In[25]:


knn.predict([[0,130,150,35,250,33.5,0.647,47]])


# In[ ]:




