import matplotlib.pyplot as plt
import os

########################### Moon
from sklearn.datasets import make_moons

def MOON_GetDataSet(n_samples):
    X,y = make_moons(n_samples= n_samples, noise = 0.05)
    return X,y

def MOON_Plot(X, y):
     plt.scatter(X[:,0], X[:,1], s=40, c=y)      

########################### MNIST
from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home

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

def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    plt.imshow(image)
    plt.axis("off")
    plt.show

########################### Iris
from sklearn.datasets import load_iris

def IRIS_GetDataSet():
    data = load_iris()
    return(data["data"], data["target"])

def IRIS_Plot(X, y):
    plt.title('Iris Data (purple=setona, teal=versicolor, yellow=virginica')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.scatter(X[:, 0], X[:, 1], c=y)
        
######################REEEEEEEEEEEEEEEEEEEEEE############
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from io import StringIO
#import warnings
from sklearn.base import clone
import numpy as np
import sys
#warnings.filterwarnings("ignore")
    
X, y = MNIST_GetDataSet()

if X.ndim==3:
    print("reshaping X..")
    assert y.ndim==1
    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
assert X.ndim==2

some_digit = X[36000]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#old_stdout = sys.stdout
#sys.stdout = mystdout = StringIO()

sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train_5)

false_list = np.array([])
true_list = np.array([])
false_list.resize((X.size,1))
true_list.resize((X.size,1))
print(X.size)

#for i in range(len(y)):
#    re = X[i]
#    temp = sgd_clf.predict([re])
#    true_list[i] = temp
##    if temp:
##        true_list[i] = temp
##    else:
##        false_list[i] = temp
#        
#plt.plot(true_list)
#plt.show

#sgd_clf.predict([some_digit])

#sys.stdout = old_stdout
#loss_history = mystdout.getvalue()
#
#loss_list = []
#
#for line in loss_history.split('\n'):
#    if(len(line.split("loss: ")) == 1):
#        continue
#    loss_list.append(float(line.split("loss: ")[-1]))
#    
#    plt.figure()
#plt.plot(np.arange(len(loss_list)), loss_list)
#plt.xlabel("Time in epochs"); plt.ylabel("Loss")
#plt.show()

#skfolds = StratifiedKFold(n_splits=3, random_state=42)
#for train_index, test_index in skfolds.split(X_train, y_train_5):
#    clone_clf = clone(sgd_clf)
#    X_train_folds = X_train[train_index]
#    y_train_folds = (y_train_5[train_index])
#    X_test_fold = X_train[test_index]
#    y_test_fold = (y_train_5[test_index])
#    clone_clf.fit(X_train_folds, y_train_folds)
#    y_pred = clone_clf.predict(X_test_fold)
#    n_correct = sum(y_pred == y_test_fold)
#    print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495



#plt.plot(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
#plt.show

#print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

