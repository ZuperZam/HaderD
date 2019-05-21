# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:47:40 2019

@author: ZuperZam
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
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
    
def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

X, y = MNIST_GetDataSet()

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X,y)
sgd_clf.fit(X_train, y_train)
sgd_val = cross_val_predict(sgd_clf, X_test, y_test, cv=3)
sgd_acc = accuracy_score(sgd_val, y_test)
print("sgd_acc: ", sgd_acc)


#generate noisy data
sgd_clf_noisy = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)

np.random.seed(42)
X_train_noisy = np.random.normal(X_train, 200)
X_test_noisy = np.random.normal(X_test, 200)

sgd_clf_noisy.fit(X_train_noisy, y_train)
sgd_val_noisy = cross_val_predict(sgd_clf_noisy, X_test_noisy, y_test, cv=3)
sgd_acc_noisy = accuracy_score(sgd_val_noisy, y_test)
print("sgd_acc_noisy: ", sgd_acc_noisy)

#plt.figure(figsize=(14, 8))
#plt.subplot(121)
#plot_digits(X_train[::2100])
#plt.title("Original", fontsize=16)
#plt.subplot(122)
#plot_digits(X_train_noisy[::2100])
#plt.title("Noisy", fontsize=16)

#test data improvement by compression
sgd_clf_pca = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)

pca = PCA(n_components=0.2)
X_train_reduced = pca.fit_transform(X_train_noisy)
X_test_reduced  = pca.fit_transform(X_test_noisy)
#X_recovered = pca.inverse_transform(X_reduced)

sgd_clf_pca.fit(X_train_reduced, y_train)
sgd_val_pca = cross_val_predict(sgd_clf_pca, X_test_reduced, y_test, cv=3)
sgd_acc_pca = accuracy_score(sgd_val_pca, y_test)
print("sgd_acc_pca: ", sgd_acc_pca)
#plt.figure(figsize=(14, 8))
#plt.subplot(121)
#plot_digits(X_train_noisy[::2100])
#plt.title("Noisy", fontsize=16)
#plt.subplot(122)
#plot_digits(X_recovered[::2100])
#plt.title("Improved by Compressed", fontsize=16)