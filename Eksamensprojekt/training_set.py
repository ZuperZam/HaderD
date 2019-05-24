# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:28:08 2019

@author: Bechy
"""

import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from scipy import misc
import glob
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

dataset = []
dataset_guess = []
y = []
y_guess = []
image_size = 500

POSITIVE_IMAGES_PATH = './puppy'
NEGATIVE_IMAGES_PATH = './not_shrek_v2'
POSITIVE_GUESS_IMAGES_PATH = './guess_positive_test'
NEGATIVE_GUESS_IMAGES_PATH = './guess_negative'

for filename in glob.glob(os.path.join(POSITIVE_IMAGES_PATH, '*.jpg')):
    image = plt.imread(filename)
    image_resized = misc.imresize(image, (image_size, image_size))
    dataset.append(image_resized)
    y.append(1)
for filename in glob.glob(os.path.join(NEGATIVE_IMAGES_PATH, '*.jpg')):
    image = plt.imread(filename)
    image_resized = misc.imresize(image, (image_size, image_size))
    dataset.append(image_resized)
    y.append(0)
    
for filename in glob.glob(os.path.join(POSITIVE_GUESS_IMAGES_PATH, '*.jpg')):
    image = plt.imread(filename)
    image_resized = misc.imresize(image, (image_size, image_size))
    dataset_guess.append(image_resized)
    y_guess.append(1)
for filename in glob.glob(os.path.join(NEGATIVE_GUESS_IMAGES_PATH, '*.jpg')):
    image = plt.imread(filename)
    image_resized = misc.imresize(image, (image_size, image_size))
    dataset_guess.append(image_resized)
    y_guess.append(0)

dataset = np.asarray(dataset)
dataset_guess = np.asarray(dataset_guess)
y = np.asarray(y)
y_guess = np.asarray(y_guess)

dataset = dataset.reshape(dataset.shape[0], image_size*image_size*3)
dataset_guess = dataset_guess.reshape(dataset_guess.shape[0], image_size*image_size*3)

####################################################################
sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42, shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(dataset, y)
#
sgd_clf.fit(X_train, y_train)
sgd_val = cross_val_predict(sgd_clf, X_test, y_test, cv=3)
sgd_acc = accuracy_score(sgd_val, y_test)
print("sgd_acc: ", sgd_acc)
####################################################################
sgd_clf_noisy = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42, shuffle=True)

X_train_noisy = np.random.normal(X_train, 200)
X_test_noisy = np.random.normal(X_test, 200)

sgd_clf_noisy.fit(X_train_noisy, y_train)
sgd_val_noisy = cross_val_predict(sgd_clf_noisy, X_test_noisy, y_test, cv=3)
sgd_acc_noisy = accuracy_score(sgd_val_noisy, y_test)
print("sgd_acc_noisy: ", sgd_acc_noisy)
####################################################################
np.random.seed(40)
guess_val = cross_val_predict(sgd_clf, dataset_guess, y_guess, cv=3)
guess_acc = accuracy_score(guess_val, y_guess)
print(guess_val)
print(guess_acc)
####################################################################
#sgd_clf_pca = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
#
#pca = PCA(n_components=0.2)
#X_train_reduced = pca.fit_transform(X_train_noisy)
#X_test_reduced  = pca.fit_transform(X_test_noisy)
##X_recovered = pca.inverse_transform(X_reduced)
#
#sgd_clf_pca.fit(X_train_reduced, y_train)
#sgd_val_pca = cross_val_predict(sgd_clf_pca, X_test_reduced, y_test, cv=3)
#sgd_acc_pca = accuracy_score(sgd_val_pca, y_test)
#print("sgd_acc_pca: ", sgd_acc_pca)
####################################################################



#logisticRegr = LogisticRegression(solver = 'lbfgs',max_iter = 1000, multi_class = 'multinomial')
#logisticRegr.fit(X_train, y_train)
#
#print('logisticRegr score: {}'.format(logisticRegr.score(X_train, y_test)))

