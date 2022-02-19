#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC

##Helper functions. Use when needed. 
def show_orignal_images(pixels):
	#Displaying Orignal Images
	fig, axes = plt.subplots(6, 10, figsize=(11, 7),
	                         subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(np.array(pixels)[i].reshape(64, 64), cmap='gray')
	plt.show()

def show_eigenfaces(pca):
	#Displaying Eigenfaces
	fig, axes = plt.subplots(3, 8, figsize=(9, 4),
	                         subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(pca.components_[i].reshape(64, 64), cmap='gray')
	    ax.set_title("PC " + str(i+1))
	plt.show()


## Step 1: Read dataset and visualize it.
df = pd.read_csv("face_data.csv")
#print(df.head())
labels = df["target"]
pixles = df.drop(['target'], axis =1)
#show_orignal_images(pixles)
## Step 2: Split Dataset into training and testing

X_train, X_test, Y_train, Y_test = train_test_split(pixles,labels)
pca = PCA(n_components = 200).fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()
#show_eigenfaces(pca)
#make tarin data in the pca word
x_train_pca= pca.transform(X_train)
## Step 3: Perform PCA.

## Step 4: Project Training data to PCA

##############

## Step 5: Initialize Classifer and fit training data
clf = SVC(kernel='rbf', C=1000, gamma=0.01)
clf.fit(x_train_pca, Y_train)
x_test_pca= pca.transform(X_test)
y_pred = clf.predict(x_test_pca)

## Step 6: Perform testing and get classification report

print(classification_report(Y_test, y_pred))


# In[33]:


from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))


# In[ ]:


s

