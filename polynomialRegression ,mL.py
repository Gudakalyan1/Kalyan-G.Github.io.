#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as metrics


# In[2]:


df=pd.read_csv("Downloads\Hr data.csv")


# In[3]:


df


# In[4]:


df.corr()


# In[38]:


x=df.iloc[:,1:2].values
y=df.iloc[:,2].values
x


# In[35]:


#checking for linear relation
plt.scatter(x,y)
plt.ylim(0,)
plt.show()


# In[39]:


#splitting data set into train set and test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train,x_test,y_train,y_test
                            


# In[40]:


#fitting linear regression to the data set
reg=linear_model.LinearRegression()
reg.fit(x,y)
reg.predict(x)


# In[9]:


#visualizing the results
plt.scatter(x,y,color='red')
plt.plot(x,reg.predict(x),color='blue')
plt.show()


# In[41]:


#fitting polynomial regression to the data set
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)


# In[66]:


pr=linear_model.LinearRegression()
pr.fit(x_poly,y)
p=pr.predict(x_poly)


# In[31]:


#visualizing the results
plt.scatter(x,y)
plt.plot(x,pr.predict(x_poly))
plt.show()


# In[71]:


reg.predict([[5.5]])


# In[72]:


#metrics calculation
mse=metrics.mean_squared_error(y,p)
r2=metrics.r2_score(y,p)
print(mse,r2)


# In[ ]:





# In[21]:





# In[ ]:




