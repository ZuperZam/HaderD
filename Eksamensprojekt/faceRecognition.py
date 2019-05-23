# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:37:54 2019

@author: Bechy
"""
import cv2
import numpy as np
import dlib
from matplotlib import pyplot as plt
import scipy.ndimage
import glob

def rotateSunglasses(degrees):
    txtfilesPNG = []
    for file in glob.glob("hidden/thug_life_sunglasses.png"):
        txtfilesPNG.append(file)
        
    for px in txtfilesPNG:
        png = plt.imread(px)
        
        return scipy.ndimage.rotate(png,degrees)


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])
    return img

txtfiles = [] 
for file in glob.glob("*.png"):
    txtfiles.append(file)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
win = dlib.image_window()
#win = dlib

for ix in txtfiles:
    img = dlib.load_rgb_image(ix)
    
    win.clear_overlay()
    win.set_image(img)
    
    dets = detector(img, 1)
    
    for k, d in enumerate(dets):
        
        shape = predictor(img, d)
        
        win.add_overlay(shape)
    win.add_overlay(dets)
    
#    img = plt.imread(ix)
    s_img = rotateSunglasses(0)
    img = overlay_image_alpha(img, s_img[:,:,0:3], [0,0], s_img[:,:,3]/255)
    win.set_image(img)











#for ix in txtfiles:
#    imgColor = plt.imread(ix)
#    img = cv2.imread(ix,cv2.IMREAD_COLOR)
#    imgtest1 = img.copy()
##    plt.imshow(imgColor)
#    imgtest = cv2.cvtColor(imgtest1, cv2.COLOR_BGR2GRAY)
#    face_cascade = cv2.CascadeClassifier('C:/Users/ZuperZam/AppData/Roaming/Python/Python36/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
#    eye_cascade = cv2.CascadeClassifier('C:/Users/ZuperZam/AppData/Roaming/Python/Python36/site-packages/cv2/data/haarcascade_eye.xml')
#    smile_cascade = cv2.CascadeClassifier('C:/Users/ZuperZam/AppData/Roaming/Python/Python36/site-packages/cv2/data/haarcascade_smile.xml')
#    
#    faces = face_cascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=2)
#
#    print('Total number of Faces found',len(faces))
#    
#    for (x, y, w, h) in faces:
#        face_detect = cv2.rectangle(imgtest, (x, y), (x+w, y+h), (255, 0, 255), 2)
#        roi_gray = imgtest[y:y+h, x:x+w]
#        roi_color = imgColor[y:y+h, x:x+w]        
##        plt.imshow(face_detect)
#        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
#        print('Total number of Eyes found',len(eyes))
#        index = 1
#        if len(eyes) >= 2 & len(eyes) < 5:
#            for (ex,ey,ew,eh) in eyes:
#                eye_detect = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
##                if index == len(eyes):
##                    plt.figure()
##                    plt.imshow(eye_detect)
##                index += 1
#        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
#        print('Total number of Smiles found', len(smile))
#        if len(smile) == 1:
#            for (ex,ey,ew,eh) in smile:
#                smile_detect = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
#                plt.figure()
#                plt.imshow(smile_detect)
#                
#    plt.show