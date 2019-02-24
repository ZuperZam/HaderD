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
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
#import warnings
import numpy as np
#warnings.filterwarnings("ignore")
    
##################Qa

X, y = MNIST_GetDataSet()

some_digit = X[36000]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train_5)

true_list = np.array([])
true_list.resize((len(X_train),1))

true = 0
false = 0

for i in range(len(y_train)):
    re = X_train[i]
    temp = sgd_clf.predict([re])
    if temp:
        true += 1
        true_list[i] = 1
    else:
        false += 1
        true_list[i] = 0

print("SDG_true = ", true)
print("SDG_false = ", false)

plt.subplot(2,1,1)
plt.plot(true_list)
plt.show

print("Predict sgd = ", cross_val_predict(sgd_clf, X_train, y_train_5, cv=3))
print("Score sgd = ", cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

##############Qb

for i in range(2):
    print()

class DummyClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print("Predict dummy = ", cross_val_predict(dummy_clf, X_train, y_train_5, cv=3))
print("Score dummy = ", cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

true_dummy = 0
false_dummy = 0

for i in range(len(y_train)):
    if y_train[i] == 5:
        true_dummy += 1
    else:
        false_dummy += 1

print("y_train_5.shape = ", y_train_5.shape[0])
print("dummy_true = ", true_dummy)
print("dummy_false = ", false_dummy)

print(confusion_matrix(y_train_5, cross_val_predict(dummy_clf, X_train, y_train_5, cv=3)))