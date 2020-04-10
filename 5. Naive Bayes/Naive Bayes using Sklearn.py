# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 06:06:19 2020

@author: vishu
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#import test and train dataset
train_data=pd.read_csv('train-data.csv')
test_data=pd.read_csv('test-data.csv')

#train_data.shape #to check shape of train dataset
#test_data.shape #to check the shape of test data

# Now, we need to predict the missing target variable in the test data
# target variable - Survived

# seperate the independent and target variable on training data
train_x=train_data.drop(columns=['Survived'],axis=1)
train_y=train_data['Survived']

#seprate the independent and target variable on test data
test_x=test_data.drop(columns=['Survived'],axis=1)
test_y=test_data['Survived']

model=GaussianNB()
#fir the model model with the training data
model.fit(train_x,train_y)

#predict the target on train dataset
predict_train=model.predict(train_x)

#print accuracy score
print('accuracy score on train dataset : ', accuracy_score(train_y,predict_train))

#predict the target on test dataset
predict_test=model.predict(test_x)

#print accuracy score
print('accuracy score on train dataset : ', accuracy_score(test_y,predict_test))