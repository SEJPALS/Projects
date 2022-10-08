#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv('MobilePriceClassification.csv')


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.n_cores.value_counts()


# In[9]:


df.shape


# In[143]:


df.corr()


# In[144]:


df.columns


# In[145]:


y=df['price_range']


# In[146]:


X=df[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi']]


# In[147]:


#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()


# In[148]:


#X=sc.fit_transform(X)


# In[149]:


from sklearn.model_selection import train_test_split


# In[150]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=2529)


# In[151]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[152]:


from sklearn.tree import DecisionTreeClassifier


# In[154]:


model=DecisionTreeClassifier()


# In[155]:


model.fit(X_train,y_train)


# In[157]:


y_pred=model.predict(X_test)


# In[158]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report 


# In[159]:


accuracy_score(y_test,y_pred)


# In[160]:


confusion_matrix(y_test,y_pred)


# In[161]:


print(classification_report(y_test,y_pred))


# In[162]:


from sklearn.neighbors import KNeighborsClassifier


# In[163]:


model1=KNeighborsClassifier()


# In[166]:


model1.fit(X_train,y_train)


# In[167]:


y_pred1=model1.predict(X_test)


# In[168]:


accuracy_score(y_test,y_pred1)


# In[169]:


confusion_matrix(y_test,y_pred1)


# In[170]:


print(classification_report(y_test,y_pred1))


# In[171]:


from sklearn.svm import SVC


# In[172]:


model2=SVC()


# In[173]:


model2.fit(X_train,y_train)


# In[174]:


y_pred2=model2.predict(X_test)


# In[175]:


accuracy_score(y_test,y_pred2)


# In[176]:


confusion_matrix(y_test,y_pred2)


# In[177]:


print(classification_report(y_test,y_pred2))


# In[178]:


from sklearn.linear_model import LogisticRegression


# In[179]:


model3=LogisticRegression()


# In[180]:


model3.fit(X_train,y_train)


# In[181]:


y_pred3=model3.predict(X_test)


# In[182]:


accuracy_score(y_test,y_pred3)


# In[183]:


confusion_matrix(y_test,y_pred3)


# In[184]:


print(classification_report(y_test,y_pred3))


# In[ ]:




