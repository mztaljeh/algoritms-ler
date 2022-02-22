#!/usr/bin/env python
# coding: utf-8

# In[2]:


#bring cancer data
from sklearn import datasets
cancer = datasets.load_breast_cancer()


# In[3]:


#print featur data
print("feature data", cancer.feature_names)


# In[4]:


#print lable data
print("feature data", cancer.target_names)


# In[5]:


#demintion of data
print(cancer.data.shape)


# In[7]:


#divide data to train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=34)


# In[10]:


#import svm and bild module
from sklearn import svm
classifer= svm.SVC(kernel='linear')
#taring data
classifer.fit(x_train,y_train)


# In[11]:


#apply module on test data
y_pred= classifer.predict(x_test)


# In[12]:


#find accurcy for module
from sklearn import metrics
print("accurcy",metrics.accuracy_score(y_test,y_pred))


# In[ ]:




