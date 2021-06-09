#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,chi2


# In[2]:


#load data set
df=pd.read_csv("Downloads\melborne_housing data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isna().sum()


# In[6]:


df.describe()


# In[7]:


df.Rooms.unique()


# In[8]:


df.Rooms.value_counts()


# In[9]:


df.Bathroom.value_counts()


# In[10]:


sns.countplot('Car',data=df)
plt.show()


# In[11]:


df.corr()


# In[12]:


melborn=df.copy()


# In[13]:


melborn.head()


# In[14]:


melborn.isna().sum()


# In[15]:


melborn['Car'].fillna(0,axis=0,inplace=True)


# In[16]:


melborn.BuildingArea.mean()


# In[17]:


melborn.BuildingArea.std()


# In[18]:


melborn[melborn['BuildingArea']>6791].index


# In[19]:


melborn.drop(index=13245,axis=0,inplace=True)


# In[20]:


melborn['BuildingArea'].fillna(melborn.BuildingArea.mean(),axis=0,inplace=True)


# In[21]:


melborn['YearBuilt'].mode()


# In[22]:


melborn.YearBuilt.value_counts()


# In[23]:


#one hot encoding
melborn.corr()


# In[24]:


sns.heatmap(melborn.corr())


# In[25]:


melborn.Regionname.unique()


# In[26]:


for col in melborn.columns:
    print(col,":" ,len(melborn[col].unique()))


# In[27]:


top15=[x for x in melborn.CouncilArea.value_counts().head(15).index]
top15


# In[28]:


for label in top15:
    melborn[label]=np.where(melborn['CouncilArea']==label,1,0)
    


# In[29]:


melborn[['CouncilArea']+top15].head(10)


# In[30]:


melborn.Suburb.value_counts().head(20)


# In[31]:


def one_hot_top_x(df,variable,top_x_lables):
    
    for label in top_x_lables:
        df[variable+'_'+label]=np.where(melborn[variable]==label, 0,1)


# In[32]:


suburb_15=[x for x in melborn.Suburb.value_counts().head(15).index]
suburb_15


# In[33]:


one_hot_top_x(melborn,'Suburb',suburb_15)


# In[34]:


melborn.head()


# In[35]:


df.SellerG.value_counts().head(15)


# In[36]:


SellerG_15=[x for x in melborn.SellerG.value_counts().head(15).index]
SellerG_15


# In[37]:


one_hot_top_x(melborn,'SellerG',SellerG_15)


# In[38]:


melborn.Propertycount


# In[39]:


df.corr()


# In[40]:


melborn.drop(['Suburb','CouncilArea','SellerG'],axis=1,inplace=True)


# In[41]:


melborn.drop(['Address','Lattitude','Longtitude','Regionname','Propertycount'],axis=1,inplace=True)


# In[42]:



melborn=pd.get_dummies(melborn,columns=['Method','Type'],drop_first=True)


# In[43]:


melborn.drop('Date',axis=1,inplace=True)


# In[44]:


melborn.YearBuilt.mode()


# In[45]:


melborn.YearBuilt.fillna(1970,axis=0,inplace=True)


# In[104]:


melborn.drop('Bedroom2',axis=1,inplace=True)


# In[105]:


x=melborn.drop('Price',axis=1)
y=melborn['Price']


# In[106]:


x


# In[107]:


bestfeatures=SelectKBest(score_func=chi2,k=10)
fit=bestfeatures.fit(x,y)


# In[108]:


dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x.columns)


# In[109]:


bestfeatures=pd.concat([dfcolumns,dfscores],axis=1)


# In[110]:


bestfeatures.columns=['specs','scores']


# In[111]:


print(bestfeatures.nlargest(15,"scores"))


# In[112]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[113]:


reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)


# In[114]:


reg.score(x_test,y_test)


# In[115]:


reg.score(x_train,y_train)


# In[116]:


y_pred=reg.predict(x_test)


# In[58]:


print('True values:',y_test)
print('predicted values:',y_pred)


# In[59]:


ax=sns.distplot(y_test,hist=False,color='green')
sns.distplot(y_pred,ax,hist=False,color='red')
plt.show()


# In[ ]:




