# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 06:32:39 2020

@author: vishu
"""

import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv('diabetes.csv')

labels=['low','medium','high']

for j in data.columns[:-1]:
    mean=data[j].mean()
    data[j]=data[j].replace(0,mean)
    data[j] = pd.cut(data[j],bins=len(labels),labels=labels)
    
    
#function to traverse through a deature and count the number of occurances
def count(data,colname,label,target):
    condition=(data[colname]== label) & (data['Outcome']== target)
    return len(data[condition])

#create a list to store our predictions
predicted=[]
#we will be storing our probabilities in a dictonary for east access of information
probabilities= {0:{},1:{}}

#split our dataset
#train_x,test_x=train_test_split(data, test_size=0.3,random_state=0)
#train_len= int(len(train_x))
train_percent = 70
train_len = int((train_percent*len(data))/100)
train_x = data.iloc[:train_len,:]
test_x = data.iloc[train_len+1:,:-1]
test_y = data.iloc[train_len+1:,-1]

count_0 = count(train_x,'Outcome',0,0)
count_1 = count(train_x,'Outcome',1,1)
    
prob_0 = count_0/len(train_x)
prob_1 = count_1/len(train_x)

#calculate probabailities
for col in train_x.columns[:-1]:
    probabilities[0][col]={}
    probabilities[1][col]={}
    
    for category in labels:
        count_ct_0=count(train_x,col,category,0)
        count_ct_1=count(train_x,col,category,1)
        probabilities[0][col][category] = count_ct_0 / count_0
        probabilities[1][col][category] = count_ct_1 / count_1
for row in range(0,len(test_x)):
    prod_0=prob_0
    prod_1=prob_1
    for  feature in test_x.columns:
        prod_0 *= probabilities[0][feature][test_x[feature].iloc[row]]
        prod_1 *= probabilities[1][feature][test_x[feature].iloc[row]]
        
        #Predict the outcomew
        if prod_0 > prod_1:
            predicted.append(0)
        else:
            predicted.append(1)

tp,tn,fp,fn = 0,0,0,0
#multiplied and divided by 8 because in my preedicted array each value was repeated 8 times continously
for j in range(0,int(len(predicted)/8.0)):
    if predicted[8*j] == 0:
        if test_y.iloc[j] == 0:
                tp += 1
        else:
                fp += 1
    else:
        if test_y.iloc[j] == 1:
                tn += 1
        else:
                fn += 1
print('Accuracy for training length '+str(train_percent)+'% : ',((tp+tn)/len(test_y))*100)