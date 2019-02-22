# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import os
###################################### Qa
#from sklearn.datasets import make_moons
#n_samples = 200
#
#def MOON_GetDataSet(n_samples):
#    X,y = make_moons(n_samples= n_samples, noise = 0.05)
#    return X,y
#
#def MOON_Plot(X, y):
#     plt.scatter(X[:,0], X[:,1], s=40, c=y)
#   
## TEST CODE:
#X, y=MOON_GetDataSet(n_samples = n_samples)
#print("X.shape=",X.shape,", y.shape=",y.shape)
#MOON_Plot(X,y)

###################################### Qb
#from sklearn.datasets import make_moons
#from sklearn.model_selection import train_test_split
#n_samples = 200
#
#def MOON_GetDataSet(n_samples):
#    X,y = make_moons(n_samples= n_samples, noise = 0.05)
#    return X,y
#
#def MOON_Plot(X, y, title="my title", xlabel="", ylabel=""):
#     split = int(n_samples / 100 * 80)
#     plt.ylabel(ylabel)
#     plt.xlabel(xlabel)
#     plt.title(title)
#     
#     plt.scatter(X[:,0], X[:,1], s=40, c=y)
#   
## TEST CODE:
#X, y=MOON_GetDataSet(n_samples = n_samples)
#print("X.shape=",X.shape,", y.shape=",y.shape)
#X_train, X_test, y_train, y_test = train_test_split(X,y)
#MOON_Plot(X_test,y_test,"test (train_test_split)","x-axis","y-axis")

###################################### Qc
#from shutil import copyfileobj
#from six.moves import urllib
#from sklearn.datasets.base import get_data_home
#from sklearn.model_selection import train_test_split
#import os
#
#def fetch_mnist(data_home=None):
#    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
#    data_home = get_data_home(data_home=data_home)
#    data_home = os.path.join(data_home, 'mldata')
#    if not os.path.exists(data_home):
#        os.makedirs(data_home)
#    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
#    if not os.path.exists(mnist_save_path):
#        mnist_url = urllib.request.urlopen(mnist_alternative_url)
#        with open(mnist_save_path, "wb") as matlab_file:
#            copyfileobj(mnist_url, matlab_file)
#from sklearn.datasets import fetch_mldata
#
#def MNIST_PlotDigit(data):
#    image = data.reshape(28, 28)
#    plt.imshow(image)
#    plt.axis("off")
#    plt.show
#
#def MNIST_GetDataSet():
#    fetch_mnist()
#    mnist = fetch_mldata('MNIST original')
#    return(mnist["data"], mnist["target"])
#
#
## TEST CODE:
#X, y = MNIST_GetDataSet()
#print("X.shape=",X.shape, ", y.shape=",y.shape)
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=50000, shuffle=True)
#
#print("X_train.shape=",X_train.shape,", X_test.shape=",X_test.shape)
#MNIST_PlotDigit(X_train[12])

###################################### Qd
#from sklearn.datasets import load_iris
#
#def IRIS_GetDataSet():
#    data = load_iris()
#    return(data["data"], data["target"])
#    
#def IRIS_Plot(X, y):
#    plt.title('Iris Data (purple=setona, teal=versicolor, yellow=virginica')
#    plt.xlabel('Sepal length')
#    plt.ylabel('Sepal width')
#    plt.scatter(X[:, 0], X[:, 1], c=y)
#
## TEST CODE:
#X, y = IRIS_GetDataSet()
#IRIS_Plot(X, y)


################################# Qe
#from sklearn.datasets import load_iris
#import seaborn as sns
#
#def IRIS_GetDataSet():
#    data = load_iris()
#    return(data["data"], data["target"])
#    
#def IRIS_PlotPair(X, y):
#    sns.set(style="ticks")
#    df = sns.load_dataset("iris")
#    sns.pairplot(df, hue="species")
#
## TEST CODE:
#X, y = IRIS_GetDataSet()
#IRIS_PlotPair(X, y)


################################# Qg