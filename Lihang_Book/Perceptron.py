# coding: utf-8

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# class Perceptron(object):
#     def __init__(self, eta=0.01, n_iter=10):
#         self.eta = eta
#         self.n_iter = n_iter
#
#     def fit(self, X, Y):
#         self.w_ = np.zeros(1 + X.shape[1])
#         self.errors_ = []
#
#         for _ in range(self.n_iter):
#             errors = 0
#             for xi, target in zip(X, Y):
#                 update = self.eta * (target - self.predict(xi))
#                 self.w_[1:] += update * xi
#                 self.w_[0] += update
#                 errors += int(update != 0.0)
#             self.errors_.append(errors)
#         return self
#
#     def net_input(self, X):
#         """计算向量点乘"""
#         return np.dot(X, self.w_[1:]) + self.w_[0]
#
#     def predict(self, X):
#         return np.where(self.net_input(X) >= 0.0, 1, -1)


"""抽取iris数据的前100条正好是两个类别，对于特征我们抽取两类特征sepel length 和petal length"""
# df = pd.read_csv('iris.data', header=None)
#
# y = df.iloc[0:100, 4].values
# y = np.where(y == 'Iris-setosa', -1, 1)
# X = df.iloc[0:100, [0, 2]].values
#
# plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
# plt.scatter(X[50:100, 0], X[50:100, 1], color="blue",
#             marker="x", label="versicolor")
# plt.xlabel("petal length")
# plt.ylabel("sepal length")
# plt.legend(loc="upper left")
# plt.show()
#
# ppn = Perceptron(eta=0, n_iter=10)
# ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
# plt.xlabel("Epoches")
# plt.ylabel("Number of misclassification")
# plt.show()
#
#
# def plot_descision_region(X, y, classifier, resolution=0.02):
#     marker = ("s", "x", "o", "^", "v")
#     colors = ("red", "blue", "lightgreen", "gray", "cyan")
#     cmap = ListedColormap(colors[:len(np.unique(y))])
#
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#
#     xx1, xx2 = np.meshgrid(np.range(x1_min, x1_max, resolution),
#                            np.range(x2_min, x2_max, resolution))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#
#     plt.contour(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
#                     c=cmap(idx), marker=markers[idx], label=cl)
#
#
# plot_decision_region(X, y, classifier=ppn)
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upperleft')
# plt.show()
