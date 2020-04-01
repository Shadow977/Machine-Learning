# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:09:19 2020

@author: vishu
"""

import matplotlib.pyplot as plt
import numpy as np


#import dataset from csv file
dataset= np.genfromtxt('salary_data.csv' ,delimiter=',')


X= dataset[1:,0].reshape(-1,1)
ones=np.ones([X.shape[0], 1])
X=np.concatenate([ones, X],1)

Y=dataset[1: ,1].reshape(-1,1)



#plt.scatter(dataset[1: , 0].reshape(-1,1) ,Y)
#plt.show()


#set the  hypothesis parameters
alpha=0.05
iters=100000

theta=np.array([[1.0,1.0]])



#define cost function
def computeCost(X,y,theta):
    temp=np.power(((X @ theta.T)-y),2) # @ means matrix multiplication of arrays.
    return np.sum(temp) / (2 * len(X))


#defining gradient Descent
def grad(X,Y,theta,alpha,iters):
    for i in range(iters):
        theta=theta- (alpha/len(X))*np.sum((X @ theta.T - Y) * X ,axis=0)
        cost=computeCost(X,Y,theta)
    return(theta,cost)

g, cost = grad(X, Y, theta, alpha, iters)  
print(g, cost)

'''
#plot
plt.scatter(dataset[:, 0].reshape(-1,1), Y)
axes = plt.gca()
x_vals = np.array(axes.get_xlim()) 
y_vals = g[0][0] + g[0][1]* x_vals #the line equation
plt.plot(x_vals, y_vals, '--')

'''

#predict the answer.

def pred(x):
    y=25792.200019876 + (9449.96232146 *x)
    print(y)

    