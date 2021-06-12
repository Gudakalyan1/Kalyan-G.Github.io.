#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Downloads\car_evaluation.csv")


# In[3]:


df.head()


# In[ ]:


#EDA


# In[5]:


df.columns


# In[178]:


col_names = ['buying','maint','doors','persons','lug_boot','safety','class']
df.columns=col_names


# In[179]:


df.columns


# In[180]:


df.head()


# In[146]:


for col in df.columns:
    print(col,":" ,len(df[col].unique()))
    


# In[22]:


for col in df.columns:
    print(col, ":",(df[col].value_counts()))


# In[181]:


df.isna().sum()


# In[210]:


x=df.drop('class',axis=1)
y=df['class']


# In[211]:


#splitting the data set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[184]:


x_train.shape


# In[212]:


x_train


# In[213]:


x_test.shape


# In[214]:


y_train.shape


# In[215]:


encode=ce.OrdinalEncoder(cols=['buying','maint','doors','persons','lug_boot','safety'])


# In[216]:


x_train=encode.fit_transform(x_train)
x_test=encode.fit_transform(x_test)


# In[217]:


x_train.value_counts()


# In[218]:


encoder=ce.OrdinalEncoder(cols=['class'])
encoder


# In[219]:


y_train.dtypes


# In[220]:


depth=np.arange(1,9)


# In[221]:


for i,d in enumerate(depth):
    cg=DecisionTreeClassifier(criterion='gini',max_depth=d,random_state=3)
    


# In[222]:


cg


# In[223]:


fit=cg.fit(x_train,y_train)


# In[224]:


y_hat=cg.predict(x_test)


# In[225]:


accuracy=metrics.accuracy_score(y_test,y_hat)
print(accuracy)


# In[226]:


cg.score(x_test,y_test)


# In[228]:


cg.score(x_train,y_train)


# In[229]:


#visualization
plt.figure(figsize=(15,15))
tree.plot_tree(fit)
plt.show()


# In[230]:


cls_entropy=DecisionTreeClassifier(criterion='entropy',max_depth=8,random_state=0)
cls_entropy.fit(x_train,y_train)


# In[231]:


cls_entropy.score(x_test,y_test)


# In[239]:


cls_entropy.score(x_train,y_train)


# In[241]:


y_pred=cls_entropy.predict(x_test)


# In[233]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
print(confusion_matrix)


# In[132]:


for col in df.columns:
    print(col,":",df[col].unique())
    


# In[237]:


y_test


# In[236]:


cls_entropy.predict([[3,4,2,3,1,1]])


# In[207]:


print(metrics.classification_report(y_test,y_pred))


# In[ ]:




