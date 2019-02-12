# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:08:26 2019

@author: Alexa
"""

#statistics.ipynb

def MeanAndVariance(y):
    #Caculate mean
    
    return make_pair<float,float>(mean, variance)

y = np.array([1,2,3,4])

m, v = MeanAndVariance(y)

expected_m = 2.5  
expected_v_biased = 1.25 # factor 1/n
expected_v_unbiased = 1.6666666666666667 # factor 1/(n-1)

print("m=",m,", diff=", m-expected_m)
print("v=",v,", diff=", v-expected_v_biased)
print(v,np.mean(y), np.var(y)) # np.var is biased(n)