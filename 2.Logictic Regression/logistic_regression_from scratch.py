# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 06:46:10 2020

@author: vishu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import time
from sklearn.model_selection import train_test_split

init_notebook_mode(connected=True)


#logistic function
def sigmoid(X, weight):
    z=np.dot(X,weight)
    return 1/(1+np.exp(-z))


#minimizing the loss
#loss function
def loss(h,y):
    return(-y * np.log(h) - (1-y)*np.log(1-h)).mean()
#gradient descent
def gradient_desc(X,h,y):
    return np.dot(X.T,(h-y))/y.shape[0]
def update_weight_loss(weight,learning_rate,gradient):
    return weight - learning_rate * gradient


#maximum likelihood estimation
def log_likelihood(x,y,weights):
    z=np.dot(x, weights)
    ll=np.sum(y*z - np.log(1+np.exp(z)))
    return ll
def gradient_ascent(X, h, y):
    return np.dot(X.T, y - h)
def update_weight_mle(weight, learning_rate, gradient):
    return weight + learning_rate * gradient


#dataset initialization
data= pd.read_csv('data.csv')
#dataset size
print("Dataset size: \n Rows {} Columns {}".format(data.shape[0], data.shape[1]))
print("Columns and data types")
pd.DataFrame(data.dtypes).rename(columns = {0:'dtype'})
print(pd.DataFrame(data.dtypes).rename(columns = {0:'dtype'}))

#copy data
df=data.copy()

#That's a lot of columns, to simplify our experiment we will only use 2 features tenure and MonthlyCharges and the target would be Churn ofcourse.
df['class']=df['Churn'].apply(lambda x : 1 if x=="YES" else 0)

#X=df[['tenure','MonthlyCharges']].copy()
X2=df[['tenure', 'MonthlyCharges']].copy()
xTrain, xTest = train_test_split(X2, test_size = 0.2, random_state = 0)

y=df['class'].copy()

#model training
start_time=time.time()
num_iter=1000

intercept=np.ones((xTrain.shape[0], 1))
xTrain=np.concatenate((intercept, xTrain),axis=1)
theta=np.zeros(xTrain.shape[1])

for i in range(num_iter):
    h=sigmoid(xTrain, theta)
    gradient=gradient_desc(xTrain, h, y)
    theta= update_weight_loss(theta, 0.1, gradient)
    
print("training time :" + str(time.time()-start_time)+ "seconds.")
print("learning rate:{} \n Iteration: {}".format(0.1, num_iter))
result=sigmoid(xTest,theta)
f=pd.DataFrame(np.around(result, decimals=6)).join(y)
f['pred']=f[0].apply(lambda x: 0 if x<0.5 else 1)
print('accuracy:')
print(f.loc[f['pred']==f['class']].shape[0]/ f.shape[0] *100)




#using maximum likelihood
start_time = time.time()
num_iter = 100000

intercept2 = np.ones((X2.shape[0], 1))
X2 = np.concatenate((intercept2, X2), axis=1)
theta2 = np.zeros(X2.shape[1])

for i in range(num_iter):
    h2 = sigmoid(X2, theta2)
    gradient2 = gradient_ascent(X2, h2, y) #np.dot(X.T, (h - y)) / y.size
    theta2 = update_weight_mle(theta2, 0.1, gradient2)
    
print("Training time (Log Reg using MLE):" + str(time.time() - start_time) + "seconds")
print("Learning rate: {}\nIteration: {}".format(0.1, num_iter))
result2 = sigmoid(X2, theta2)
print("Accuracy (Maximum Likelihood Estimation):")
f2 = pd.DataFrame(result2).join(y)
print(f2.loc[f2[0]==f2['class']].shape[0] / f2.shape[0] * 100)



