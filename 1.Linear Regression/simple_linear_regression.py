# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:19:43 2020

@author: vishu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#importing Dataset
dataset=pd.read_csv('salary_data.csv')
X=dataset.iloc[:, :-1].values #extracting data excluding last column
Y=dataset.iloc[:, 1].values   #extreacting data in last column

X_train, X_test ,y_train ,y_test= train_test_split(X,Y,test_size=1/3 , random_state=0)

#fitting linear regression to the training set
regressor= LinearRegression()
regressor.fit(X_train,y_train)


# Visualizing the Training set results

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()


# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred) #you can pass a array to the predictor

