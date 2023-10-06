import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
y = iris.target
kf = KFold(n_splits=5, random_state=1, shuffle=True)
logreg_scores = []
clf_scores = []

print("\nFold Cross Validation Approach Logistic Regression -> ")
logreg = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    logreg_scores.append(accuracy)
    logreg_std_dev = np.std(logreg_scores)
    logreg_mean = np.mean(logreg_scores)

print("\nFold Cross Validation Approach Decision Tree Model -> ")
clf = DecisionTreeClassifier()

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    clf_scores.append(accuracy)
    clf_std_dev = np.std(clf_scores)
    clf_mean = np.mean(clf_scores)

print('Standard Deviation of Accuracy (Logistic Regression): {:.2f}'.format(logreg_std_dev))
print('Standard Deviation of Accuracy (Decision Tree): {:.2f}'.format(clf_std_dev))
print('Mean Accuracy (Logistic Regression): {:.2f}'.format(logreg_mean))
print('Mean Accuracy (Decision Tree): {:.2f}'.format(clf_mean))

# Plot the accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), logreg_scores, marker='o', linestyle='-', color='b', label='Logistic Regression')
plt.plot(range(1, 6), clf_scores, marker='o', linestyle='-', color='g', label='Decision Tree')
plt.title("Accuracy vs. Fold (Logistic Regression vs. Decision Tree)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


