import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn import tree
print("\n Fold Cross Validation Approach Logistic Regression on the Iris DataSet -> ")
irisDataset =genfromtxt('IrisNew.csv', delimiter=',',dtype=None,encoding=None)
x = pd.DataFrame(irisDataset[1:,:4])
y = pd.DataFrame(irisDataset[1:,4]).values.flatten()
# Splitting the dataset for training and testing 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=12)

irisDatasetModel = linear_model.LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
c = KFold(n_splits=5, random_state=1, shuffle=True)
scores = cross_val_score(irisDatasetModel, x, y, scoring='accuracy', cv=c, n_jobs=-1)
# report performance
print('The mean accuracy of the model is',mean(scores))
print('The Standard Deviation accuracy of the model is',std(scores))

print("\n Fold Cross Validation Approach Decision Tree Model on the Iris DataSet -> ")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
scores = cross_val_score(clf, x, y, scoring='accuracy', cv=c, n_jobs=-1)
print('The mean accuracy of the model is',mean(scores))
print('The Standard Deviation accuracy of the model is',std(scores))



