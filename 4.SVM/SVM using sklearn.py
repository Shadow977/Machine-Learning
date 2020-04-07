# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:30:53 2020

@author: vishu
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

cancer=datasets.load_breast_cancer()
print("Features:",cancer.feature_names)
print("labels:" , cancer.target_names)

#cancer.data.shape
#print(cancer.data[0:5])
#print(cancer.target)

X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,test_size=0.25, random_state=100) #75% train data and 25% test data
clf=svm.SVC(kernel='linear') #linear kernel
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("accuracy: ", metrics.accuracy_score(y_test,y_pred))# Model Accuracy: how often is the classifier correct?
print("precision:",metrics.precision_score(y_test,y_pred))# Model Precision: what percentage of positive tuples are labeled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))# Model Recall: what percentage of positive tuples are labelled as such?