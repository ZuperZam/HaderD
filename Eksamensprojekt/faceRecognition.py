# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:37:54 2019

@author: Bechy
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

txtfiles = [] 
for file in glob.glob("*.jpg"):
    txtfiles.append(file)
    
for ix in txtfiles:
    imgColor = plt.imread(ix)
    img = cv2.imread(ix,cv2.IMREAD_COLOR)
    imgtest1 = img.copy()
    plt.imshow(imgColor)
    imgtest = cv2.cvtColor(imgtest1, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('C:/Users/ZuperZam/AppData/Roaming/Python/Python36/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:/Users/ZuperZam/AppData/Roaming/Python/Python36/site-packages/cv2/data/haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=2)

    print('Total number of Faces found',len(faces))
    
    for (x, y, w, h) in faces:
        face_detect = cv2.rectangle(imgtest, (x, y), (x+w, y+h), (255, 0, 255), 2)
        roi_gray = imgtest[y:y+h, x:x+w]
        roi_color = imgColor[y:y+h, x:x+w]        
#        plt.imshow(face_detect)
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.01, minNeighbors=10)
        print('Total number of Eyes found',len(eyes))
        index = 1
        if len(eyes) >= 2 & len(eyes) < 5:
            for (ex,ey,ew,eh) in eyes:
                eye_detect = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
                if index == len(eyes):
                    plt.figure()
                    plt.imshow(eye_detect)
                    print('ble')
                index += 1

                
    plt.show