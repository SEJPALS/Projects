#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd


# In[60]:


df=pd.read_csv('MushroomClassification.csv')


# In[61]:


df


# In[62]:


df.head()


# In[63]:


df.describe()


# In[64]:


df.nunique()


# In[65]:


df.shape


# In[66]:


df.columns


# In[67]:


y=df['class']


# In[68]:


y.shape


# In[69]:


X=df[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']]


# In[70]:


from sklearn.preprocessing import OrdinalEncoder
OE=OrdinalEncoder()


# In[71]:


X=OE.fit_transform(X)


# In[72]:


X.shape


# In[76]:


from sklearn.model_selection import train_test_split


# In[77]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=2529)


# In[78]:


X_train.shape,X_test.shape,y_test.shape,y_train.shape


# In[79]:


from sklearn.linear_model import LogisticRegression


# In[80]:


model2=LogisticRegression(max_iter=1000)


# In[81]:


model2.fit(X_train,y_train)


# In[82]:


y_pred2=model2.predict(X_test)


# In[83]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[84]:


accuracy_score(y_test,y_pred2)


# In[85]:


confusion_matrix(y_test,y_pred2)


# In[86]:


print(classification_report(y_test,y_pred2))


# In[87]:


from sklearn.neighbors import KNeighborsClassifier


# In[88]:


model=KNeighborsClassifier()


# In[89]:


model.fit(X_train,y_train)


# In[91]:


y_pred=model.predict(X_test)


# In[92]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[93]:


accuracy_score(y_test,y_pred)


# In[94]:


print(classification_report(y_test,y_pred))


# In[95]:


confusion_matrix(y_test,y_pred)


# In[96]:


from sklearn.tree import DecisionTreeClassifier


# In[97]:


model1=DecisionTreeClassifier()


# In[98]:


model1.fit(X_train,y_train)


# In[99]:


y_pred1=model1.predict(X_test)


# In[100]:


accuracy_score(y_test,y_pred1)


# In[102]:


confusion_matrix(y_test,y_pred1)


# In[103]:


print(classification_report(y_test,y_pred1))


# In[ ]:




