# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:20:10 2019

@author: ZuperZam
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
import os

def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)
            
from sklearn.datasets import fetch_mldata

def MNIST_GetDataSet():
    fetch_mnist()
    mnist = fetch_mldata('MNIST original')
    return(mnist["data"], mnist["target"])

X, y = MNIST_GetDataSet()

# Opsætning af classifiers
gaussNB = GaussianNB()
multiNB = MultinomialNB()
bernoNB = BernoulliNB()
sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)

# Train, test, split
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Fit modeller
gaussNB.fit(X_train, y_train)
multiNB.fit(X_train, y_train)
bernoNB.fit(X_train, y_train)
sgd_clf.fit(X_train, y_train)

# Beregne cross validation score
gauss_val = cross_val_predict(gaussNB, X_test, y_test, cv=3)
multi_val = cross_val_predict(multiNB, X_test, y_test, cv=3)
berno_val = cross_val_predict(bernoNB, X_test, y_test, cv=3)
sgd_val = cross_val_predict(sgd_clf, X_test, y_test, cv=3)

# Beregne accuracy
gauss_acc = accuracy_score(gauss_val, y_test)
multi_acc = accuracy_score(multi_val, y_test)
berno_acc = accuracy_score(berno_val, y_test)
sgd_acc = accuracy_score(sgd_val, y_test)

# Print
print("Gaussian Naive Bayes accuracy: ", gauss_acc,
      " - Multinomial Naive Bayes accuracy: ", multi_acc,
      " - Bernoulli Naive Bayes accuracy: ", berno_acc,
      " - Stochastic Gradient Descent accuracy: ", sgd_acc)
