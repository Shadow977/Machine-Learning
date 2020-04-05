# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:45:05 2020

@author: vishu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as sslr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score


#read data from csv
dataset=pd.read_csv('User_Data.csv')

#input
x=dataset.iloc[:,[2,3]].values #to find wether the person purchased the item or not, we find relationship b/w age and salary
#output
y=dataset.iloc[:,4].values

#spliting the dataset
xtrain,xtest,ytrain,ytest= tts(x,y,test_size=0.25,random_state=0)


#feature scaling
sc_x=sslr()
xtrain=sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
print(xtrain[0:10, :])

#traing the logistic model
classifier=LogisticRegression(random_state=0)
classifier.fit(xtrain,ytrain)

#prediction
ypred=classifier.predict(xtest)

#confusion matrix
cm=confusion_matrix(ytest,ypred)
print('Confusion matrix: \n',cm)

#accuracy
print('Accuracy: ', accuracy_score(ytest, ypred))


#visualising data
from matplotlib.colors import ListedColormap 
X_set, y_set = xtest, ytest 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,  
                               stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1,  
                               stop = X_set[:, 1].max() + 1, step = 0.01)) 
  
plt.contourf(X1, X2, classifier.predict( 
             np.array([X1.ravel(), X2.ravel()]).T).reshape( 
             X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green'))) 
  
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
  
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j) 
      
plt.title('Classifier (Test set)') 
plt.xlabel('Age') 
plt.ylabel('Estimated Salary') 
plt.legend() 
plt.show() 
