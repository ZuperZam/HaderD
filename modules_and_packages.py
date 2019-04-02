# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:49:54 2019

@author: Alexa
"""
from libitmal import __init__
from libitmal import utils as itmalutils

itmalutils.TestAll()

import sys,os
from moduletest1 import moduletest1 as test
sys.path.append(os.path.expanduser('~/itmal'))

print(dir(itmalutils))
print(itmalutils.__file__)

test.doesItWork()

# Spyder can be forced to sync up to the new modules by going to FILE > Restart (ALT + SHIFT + R)
# Alternatively, one can also use a reload function that accepts the module to be reloaded as a parameter

class MyClass:
    myvar = "blah"
    
    def __init__(self):
        self.x = "Hello"

    def pubFunction(self):
        print("This is a public function")
        
    def __privFunction(self):
        print("This is a private function")
        
    def myfun(self):
        print("This is a message inside the class.")
        self.myvar = "bleh"
        return self.myvar
    
    def __repr__(self):
        return str(self.__dict__)

a = 10
##print("The number is" + a) ## THIS WILL NOT WORK
print("The number is " + a.__repr__())

myobjectx = MyClass()

print(myobjectx.myfun())
print(myobjectx.myvar)

# The meaning of "self" in Python is used to reference instance attributes. Meaning it is used as its own object to
# reference the the object of a class. This means whenever a variable is changed, for example by doing "self.myvar",
# the "self" will reference the class object, thus making sure that only the variable inside that class changes.

# If self is not included in a function parameter, the program doesn't know what instance it has to assign any
# attribute changes to.

# Constructors in Python are declared by first creating an object using a __new__-function. To initialize the newly
# created object one uses the __init__-function to do so. This function only needs to have self as a parameter.
# One can provide additional parameters if one needs to set default values for variables. The __new__ and __init__-functions
# together form what is essentially a constructor in Python.

# Destructors are not required since Python has garbage collection