#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling  import SMOTE
from sklearn.tree import DecisionTreeClassifier


# In[2]:


df=pd.read_csv("Downloads\HR_comma_sep.csv")


# In[3]:


df.head()


# In[4]:


df.isna().sum()


# In[5]:


df.describe().T


# In[6]:


df.corr().T


# In[7]:


df.info()


# In[8]:


df.Department.unique()


# In[9]:


df.shape


# In[10]:


columns=df.columns


# In[11]:


for col in columns:
    print(col,":",len(df[col].unique()))


# In[12]:


for col in columns:
    print(col,":",df[col].value_counts())


# In[13]:


#Exploratory data analysis
cols=df.columns.tolist()
cols


# In[14]:


for i in ['number_project','salary','Department','left']:
    plt.figure(figsize=(10,5))
    sns.countplot(df[i])
    plt.show()


# In[15]:



pd.crosstab(df.Department,df.left).plot(kind='bar')
plt.show()


# In[16]:


pd.crosstab(df.salary,df.left).plot(kind='bar')


# In[17]:


#Feature engineering
df.head()


# In[18]:


hr=df[['satisfaction_level','average_montly_hours','time_spend_company','promotion_last_5years','salary']]


# In[19]:


hr.head()


# In[20]:


#ordinal Encoding
encoder=ce.OrdinalEncoder(cols=['salary'])
x=encoder.fit_transform(hr)


# In[21]:


# x matrix and y vector
x=x.values
y=df['left']


# # splitting the data set

# In[22]:



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[23]:


lr=linear_model.LogisticRegression()


# In[24]:


lr.fit(x_train,y_train)


# In[25]:


lr.score(x_test,y_test)


# In[26]:


lr.score(x_train,y_train)


# In[27]:


y_hat=lr.predict(x_test)


# In[28]:


cm=metrics.confusion_matrix(y_test,y_hat)
cm


# In[29]:


sns.heatmap(cm)


# In[30]:


print(metrics.classification_report(y_test,y_hat))


# In[31]:


df.left.value_counts()


# In[52]:


df.left.value_counts()/df.left.value_counts().sum()*100


# # oversampling

# In[40]:


smote=SMOTE(sampling_strategy='minority')


# In[47]:


x_train_sm,y_train_sm=smote.fit_resample(x_train,y_train)


# In[48]:


x_train_sm.shape


# In[50]:


y_train_sm.shape


# In[53]:


log=linear_model.LogisticRegression()
log.fit(x_train_sm,y_train_sm)


# In[54]:


log.score(x_test,y_test)


# In[55]:


log.score(x_train_sm,y_train_sm)


# In[56]:


y_predic=log.predict(x_test)


# In[58]:


print('classification report:','\n',metrics.classification_report(y_test,y_predic))


# In[61]:


dt=DecisionTreeClassifier(max_depth=3,random_state=3)
dt.fit(x_train_sm,y_train_sm)


# In[62]:


dt.score(x_test,y_test)


# In[63]:


y_hat1=dt.predict(x_test)


# In[70]:


print('test data:','\n',metrics.classification_report(y_test,y_hat1))


# In[ ]:




