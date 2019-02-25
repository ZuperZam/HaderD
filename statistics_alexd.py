# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:08:26 2019
@author: Alexa
"""

#statistics.ipynb

#////////////////////////////////////////////#
#/////////////////Statistics Qa//////////////#
#////////////////////////////////////////////#

import statistics
import numpy as np

def MeanAndVariance(y):
    mean = statistics.mean(y)
    variance = statistics.variance(y)
    #Caculate mean
    return (mean, variance)

y = np.array([1,2,3,4])

m, v = MeanAndVariance(y)

expected_m = 2.5  
expected_v_biased = 1.25 # factor 1/n
expected_v_unbiased = 1.6666666666666667 # factor 1/(n-1)

print("m=",m,", diff=", m-expected_m)
print("v=",v,", diff=", v-expected_v_biased)
print(v,np.mean(y), np.var(y)) # np.var is biased(n)


#////////////////////////////////////////////#
#/////////////////Statistics Qb//////////////#
#////////////////////////////////////////////#

import numpy as np

Xi = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
N = np.size(Xi)
k = 5
Xs = np.average(Xi)

def AutoCovarianceMatrix(Xi, N, k, Xs):
    autoCov = 0
    for i in np.arange(0, N - k):
        autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
    return (1/(N-1))*autoCov

print("Autovariance:", AutoCovarianceMatrix(Xi, N, k, Xs))
print("Autovariance test:", np.cov(Xi, None, True))


#////////////////////////////////////////////#
#/////////////////Statistics Qc//////////////#
#////////////////////////////////////////////#

from scipy.stats import pearsonr
import numpy as np

x = np.array([23, 12, 45, 67, 90])
y = np.array([14, 32, 76, 89, 29])

print(np.sum(x))
print(np.sum(y))

def PearsonR(x, y):
    pearsonR = (np.cov(x, y))/(MeanAndVariance(x)*MeanAndVariance(y))
    return pearsonR

print(PearsonR(x, y))

r, p = pearsonr(x, y)

print("r", r)
print("p", p)