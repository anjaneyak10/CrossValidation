import pandas as pd
from sklearn import linear_model
from numpy import genfromtxt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn import tree
import matplotlib.pyplot as plt


print("\n Fold Cross Validation Approach Logistic Regression on the Iris DataSet -> ")
irisDataset =genfromtxt('IrisNew.csv', delimiter=',',dtype=None,encoding=None)
x = pd.DataFrame(irisDataset[1:,:4])
y = pd.DataFrame(irisDataset[1:,4]).values.flatten()



lrModel = linear_model.LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
c = KFold(n_splits=5, random_state=2, shuffle=True)
scores = cross_val_score(lrModel, x, y, scoring='accuracy', cv=c, n_jobs=-1)
print(scores)
plt.bar(range(1, 6), scores, color ='maroon', width = 0.4)
print('The mean accuracy of the model is',mean(scores))
print('The Standard Deviation accuracy of the model is',std(scores))
print("\nFold Cross Validation Approach Decision Tree Model on the Iris DataSet -> ")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(min(scores)-.05,1)
plt.axhline(mean(scores))
plt.show()



clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf, x, y, scoring='accuracy', cv=c, n_jobs=-1)
plt.bar(range(1, 6), scores, color ='maroon', width = 0.4)
print(scores)
print('The mean accuracy of the model is',mean(scores))
print('The Standard Deviation accuracy of the model is',std(scores))
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(min(scores)-.05,1)
plt.axhline(mean(scores))
plt.show()
