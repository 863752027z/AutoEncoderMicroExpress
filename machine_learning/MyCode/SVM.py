import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets

iris = datasets.load_iris()
X = iris['data'][:, (2, 3)]
print(X.shape)
y = iris['target']
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

print(X.shape)
print(y.shape)
svm_clf = SVC(kernel='linear', C=float('inf'))
svm_clf.fit(X, y)


def plot_svc_decision_boundary(X, y, svm_clf):
    w = svm_clf.coef_
    b = svm_clf.intercept_
    x0 = np.linspace(0, 5.5, 200)
    decision_boundary = -w[0][0]/w[0][1]*x0 -b/w[0][1]
    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, 'k-', linewidth=2)
    print()
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'ys')
    plt.show()

#plot_svc_decision_boundary(X, y, svm_clf)
predict = svm_clf.predict([[1, 1], [2, 2]])
print(predict)