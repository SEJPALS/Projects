#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


df=pd.read_csv('HeartDisease.csv')


# In[5]:


df


# In[6]:


df.head()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df.nunique()


# In[11]:


df.corr()


# In[12]:


df.shape


# In[13]:


df.columns


# In[14]:


y=df['target']


# In[16]:


y.shape


# In[15]:


X=df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']]


# In[17]:


X.shape


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=2529)


# In[21]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[22]:


from sklearn.neighbors import KNeighborsClassifier


# In[23]:


model=KNeighborsClassifier()


# In[25]:


model.fit(X_train,y_train)


# In[26]:


y_pred=model.predict(X_test)


# In[27]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[28]:


accuracy_score(y_test,y_pred)


# In[29]:


confusion_matrix(y_test,y_pred)


# In[30]:


print(classification_report(y_test,y_pred))


# In[31]:


from sklearn.tree import DecisionTreeClassifier


# In[32]:


model1=DecisionTreeClassifier()


# In[33]:


model1.fit(X_train,y_train)


# In[34]:


y_pred1=model1.predict(X_test)


# In[35]:


accuracy_score(y_test,y_pred1)


# In[37]:


print(classification_report(y_test,y_pred1))


# In[38]:


confusion_matrix(y_test,y_pred1)


# In[39]:


from sklearn.linear_model import LogisticRegression


# In[49]:


model2=LogisticRegression(max_iter=1000)


# In[50]:


model2.fit(X_train,y_train)


# In[51]:


y_pred2=model2.predict(X_test)


# In[57]:


accuracy_score(y_test,y_pred2)


# In[56]:


confusion_matrix(y_test,y_pred2)


# In[55]:


print(classification_report(y_test,y_pred2))


# In[ ]:




