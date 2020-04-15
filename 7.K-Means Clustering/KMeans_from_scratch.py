# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 06:21:34 2020

@author: vishu
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

dataset=pd.read_csv('Mall_Customers.csv')
#dataset.describe()

#for viualization convenience,im going to take annual income and spending score as our data
X=dataset.iloc[:,[3,4]].values
m=X.shape[0]#number of training examples
n=X.shape[1] #number of features. here n=2
n_iter=100 #number of iterations to converge
k=5 #number of clusters

centroids=np.array([]).reshape(n,0)
for i in range(k):
    rand=random.randint(0,m-1)
    centroids=np.c_[centroids,X[rand]]
    
output={}
for i in range(n_iter):
     #step 2.a
      EuclidianDistance=np.array([]).reshape(m,0)
      for k in range(k):
          tempDist=np.sum((X-centroids[:,k])**2,axis=1)
          EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      C=np.argmin(EuclidianDistance,axis=1)+1
     #step 2.b
      Y={}
      for k in range(k):
          Y[k+1]=np.array([]).reshape(2,0)
      for i in range(m):
          Y[C[i]]=np.c_[Y[C[i]],X[i]]
     
      for k in range(k):
          Y[k+1]=Y[k+1].T

      for k in range(k):
          centroids[:,k]=np.mean(Y[k+1],axis=0)
      output=Y
         
         
#visualise unclustered data
plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.title('Plot of data points')
plt.show()


#Now let’s plot the clustered data:

color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(k):
    plt.scatter(output[k+1][:,0],output[k+1][:,1],c=color[k],label=labels[k])
    plt.scatter(centroids[0,:],centroids[1,:],s=300,c='yellow',label='Centroids')
    plt.xlabel('Income')
    plt.ylabel('Number of transactions')
    plt.legend()
    plt.show()
