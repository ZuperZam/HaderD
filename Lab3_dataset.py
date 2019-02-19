# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

###################################### Qa
#import matplotlib.pyplot as plt
#import numpy as np
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
#import matplotlib.pyplot as plt
#import numpy as np
#from sklearn.datasets import make_moons
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
#     X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
#     
#     plt.scatter(X[:,0], X[:,1], s=40, c=y)
#   
## TEST CODE:
#X, y=MOON_GetDataSet(n_samples = n_samples)
#print("X.shape=",X.shape,", y.shape=",y.shape)
#MOON_Plot(X,y,"train (train_test_split)","x-axis","y-axis")

###################################### Qc
from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
from sklearn.model_selection import train_test_split
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

def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    plt.imshow(image)
    plt.axis("off")
    plt.show

def MNIST_GetDataSet():
    fetch_mnist()
    mnist = fetch_mldata('MNIST original')
    return(mnist["data"], mnist["target"])


# TEST CODE:
X, y = MNIST_GetDataSet()
print("X.shape=",X.shape, ", y.shape=",y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=50000, shuffle=True)

print("X_train.shape=",X_train.shape,", X_test.shape=",X_test.shape)
MNIST_PlotDigit(X_train[12])

###################################### Qd
#from shutil import copyfileobj
#from six.moves import urllib
#from sklearn.datasets.base import get_data_home
#from sklearn.model_selection import train_test_split
#import sklearn
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
#def IRIS_GetDataSet():
#    fetch_mnist()
#    mnist = sklearn.datasets.load_iris()
#    return(mnist["data"], mnist["target"])
#
## TEST CODE:
#X, y = IRIS_GetDataSet()
#print("X.shape=",X.shape, ", y.shape=",y.shape)
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100, shuffle=True)
#
#print("X_train.shape=",X_train.shape,", X_test.shape=",X_test.shape)
#MNIST_PlotDigit(X_train[12])

################################# Qe
#from shutil import copyfileobj
#from six.moves import urllib
#from sklearn.datasets.base import get_data_home
#from sklearn.model_selection import train_test_split
#import sklearn
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
#def IRIS_GetDataSet():
#    fetch_mnist()
#    mnist = sklearn.datasets.load_iris()
#    return(mnist["data"], mnist["target"])
#    
#def plotFunc(X,y):
#    import seaborn as sns
#    sns.set(style="ticks")
#
#    df = sns.load_dataset("iris")
#    sns.pairplot(x_vars=X, y_vars=y, hue="species")
#
## TEST CODE:
#X, y = IRIS_GetDataSet()
#print("X.shape=",X.shape, ", y.shape=",y.shape)
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100, shuffle=True)
#
#print("X_train.shape=",X_train.shape,", X_test.shape=",X_test.shape)
##MNIST_PlotDigit(X_train[12])
#plotFunc(X,y)



    







