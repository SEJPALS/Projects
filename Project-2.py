#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/ybifoundation/Dataset/main/Diabetes.csv')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[8]:


df.nunique()


# In[9]:


df.corr()


# In[7]:


df.shape


# In[10]:


df.columns


# In[51]:


y=df['diabetes']


# In[52]:


y.shape


# In[53]:


X=df[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
       'dpf', 'age']]


# In[54]:


X.shape


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=2529)


# In[57]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[34]:


from sklearn.neighbors import KNeighborsClassifier


# In[35]:


model=KNeighborsClassifier()


# In[36]:


model.fit(X_train,y_train)


# In[37]:


y_pred=model.predict(X_test)


# In[38]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[39]:


accuracy_score(y_test,y_pred)


# In[40]:


confusion_matrix(y_test,y_pred)


# In[41]:


print(classification_report(y_test,y_pred))


# In[63]:


from sklearn.linear_model import LogisticRegression


# In[64]:


model1=LogisticRegression()


# In[67]:


model1.fit(X_train,y_train)


# In[68]:


y_pred=model1.predict(X_test)


# In[69]:


accuracy_score(y_test,y_pred)


# In[70]:


confusion_matrix(y_test,y_pred)


# In[71]:


print(classification_report(y_test,y_pred))


# In[ ]:




