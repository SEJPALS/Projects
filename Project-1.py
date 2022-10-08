#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('BankNoteAuthentication.csv')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.nunique()


# In[8]:


df.corr()


# In[9]:


df.columns


# In[10]:


df.shape


# In[11]:


y=df['class']


# In[12]:


y.shape


# In[13]:


X=df[['variance', 'skewness', 'curtosis', 'entropy']]


# In[14]:


X.shape


# In[15]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=2529)


# In[16]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


model=LogisticRegression()


# In[19]:


model.fit(X_train,y_train)


# In[20]:


y_pred=model.predict(X_test)


# In[21]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[22]:


accuracy_score(y_test,y_pred)


# In[23]:


confusion_matrix(y_test,y_pred)


# In[24]:


print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:




