# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:39:29 2020

@author: vishu
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#importing dataset
train_data=pd.read_csv('train-data.csv')
test_data=pd.read_csv('test-data.csv')

#shape of dataset
print('shape of training data: ', train_data.shape)
print('shape of test data:',test_data.shape)

#we need to predict the missing target variable in the test data
#target variable- survived
#seprate the independent and target variable on training data
train_x= train_data.drop(columns=['Survived'],axis=1)
train_y=train_data['Survived']
#seprate the independent and target variable on test data
test_x = test_data.drop(columns=['Survived'],axis=1)
test_y = test_data['Survived']

#create the object of decision tree model, you can add other parameters too..
model=DecisionTreeClassifier()

model.fit(train_x,train_y)
print('tree depth:',model.get_depth())

predict_train=model.predict(train_x)
print('target on train data',predict_train)

accuracy_train=accuracy_score(train_y,predict_train)
print(accuracy_train)


predict_test=model.predict(test_x)
print('target on test data:',predict_test)

accuracy_test=accuracy_score(test_y,predict_test)
print(accuracy_test)