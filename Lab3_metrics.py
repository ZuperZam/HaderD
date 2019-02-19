# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:19:40 2019

@author: Bechy
"""

def MyAccuracy(y_pred, y_true):
    # TODO: you impl here
    


# TEST FUNCTION: compare with Scikit-learn accuracy_score
def TestAccuracy(y_pred, y_true):
    a0=MyAccuracy(y_pred, y_true)
    a1=accuracy_score(y_pred, y_true)

    print("\nmy a          =",a0)
    print("scikit-learn a=",a1)

    itmalutils.InRange(a0,a1)