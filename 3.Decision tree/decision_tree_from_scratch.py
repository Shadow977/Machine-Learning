# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:36:52 2020

@author: vishu
"""
from __future__ import print_function

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

#find unique values for a col in dataset
def unique_vals(rows,col):
    return set([row[col] for row in rows])

#counts the number of each type of example in dataset
def class_counts(rows):
    counts={} #dictionary of label- count
    for row in rows:
        label=row[-1] #label is always the last col in our dataset
        if label not in counts:
            counts[label]=0
        counts[label] +=1
    return counts

#test if value is numeric
def is_num(value):
    return isinstance(value,int) or isinstance(value,float)

'''
  This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
'''

class Question:
    def __init__(self, column, value):
        self.column=column
        self.value=value
        
    # Compare the feature value in an example to the
    # feature value in this question.
    def match(self , example):
        val=example[self.column]
        if is_num(val):
            return val >=self.value
        else:
            return val == self.value
        
        # This is just a helper method to print
        # the question in a readable format.
    def __repr__(self):
        condition = "=="
        if is_num(self.value):
            condition = ">="
        return "Is %s %s %s?" % ([self.column],condition,str(self.value))
    
    
"""
    Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
"""    
def partitions(rows,question):
    true_rows, false_rows=[],[]
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

#defining gini index
def gini(rows):
    counts= class_counts(rows)
    impurity=1
    for lbl in counts:
        prob_lbl=counts[lbl]/float(len(rows))
        impurity-=prob_lbl**2
    return impurity

#information gain(unceertainity of starting node minus the weight of impurity of two child nodes)
def info_gain(left,right,current_uncertainity):
    p=float(len(left)) / (len(left) + len(right))
    return current_uncertainity- p*gini(left) -(1-p)*gini(right)

"""
    Find the best question to ask by iterating over every feature / value
    and calculating the information gain.
"""
def find_best_split(rows):
    best_gain=0     #keep track of best info gain
    best_question=None  #keep track of features that produced it 
    current_uncertainity=gini(rows)
    n_features=len(rows[0])-1 #number of colums
    
    for col in range(n_features):
        values=set([row[col] for row in rows]) #unique values in the column
        for val in values:
            question=Question(col,val)
            true_rows, false_rows=partitions(rows,question) #try spliting dataset
            if len(true_rows)==0 or len(false_rows)==0:
                continue #skip the split if it doesn't divide the dataset
            
            gain=info_gain(true_rows,false_rows,current_uncertainity) #calculate information gain from the split
            #comparing and updating the best_gain and best_question
            if gain >= best_gain:
                best_gain ,best_question= gain, question
    return best_gain,best_question


class Leaf:
    def __init__(self,rows):
        self.predictions=class_counts(rows)
        
class Decision_Node():
    def __init__(self,question,true_branch,false_branch):
        self.question=question
        self.true_branch=true_branch
        self.false_branch=false_branch
        
def build_tree(rows):
    gain, question=find_best_split(rows)
    if gain==0:
        return Leaf(rows) #if gain=0, since we can ask no further questions, we'll return a leaf.
    true_rows , false_rows = partitions(rows,question)
    true_branch=build_tree(true_rows) #recursively build the true branch
    false_branch=build_tree(false_rows) #recursively build the false branch
    return Decision_Node(question,true_branch,false_branch) #returns a question node. this records the best feature/value to ask at this point, as well as the branches to follow depending on the answer.


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")
    
def classify(row,node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row,node.true_branch)
    else:
        return classify(row, node.false_branch)
    
def print_leaf(counts):
    total=sum(counts.values())* 1.0
    probs={}
    for lbl in counts.keys():
        probs[lbl]=str(int(counts[lbl]/total * 100)) + "%"
    return probs
    

if __name__ == '__main__':

    my_tree = build_tree(training_data)

    print_tree(my_tree)

    # Evaluate
    testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

    for row in testing_data:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify(row, my_tree))))