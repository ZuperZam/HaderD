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
import glob
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = []
y = []

POSITIVE_IMAGES_PATH = './shrek'
NEGATIVE_IMAGES_PATH = './not_shrek_v2'

for filename in glob.glob(os.path.join(POSITIVE_IMAGES_PATH, '*.jpg')):
    image = cv2.imread(filename)
    dataset.append([image,1])
    y.append('1')
for filename in glob.glob(os.path.join(NEGATIVE_IMAGES_PATH, '*.jpg')):
    image = cv2.imread(filename)
    dataset.append([image,1])
    y.append('0')
    
sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(dataset, y)

sgd_clf.fit(X_train, y_train)
#sgd_val = cross_val_predict(sgd_clf, X_test, y_test, cv=3)
#sgd_acc = accuracy_score(sgd_val, y_test)



#logisticRegr = LogisticRegression(solver = 'lbfgs',max_iter = 1000, multi_class = 'multinomial')
#logisticRegr.fit(X_train, y_train)

#print('logisticRegr score: {}'.format(logisticRegr.score(X_train, y_test)))

