# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:41:17 2022

@author: ThisAintPorn
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('peter.jpg')
#______________________________________________Test
cv2.imshow('First image check', img)

iWidth = img.shape[1]
iHeight = img.shape[0]

#Scaling percentage for resizing image if too big
scale_factor = 1.00

#Implement state machine?


#Resizing image with the scale percentage
img = cv2.resize(img,(int(iWidth * scale_factor), int(iHeight * scale_factor)))

#Grayscaling
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#______________________________________________Test
cv2.imshow('Grayscale', img_gray)

#Creating histogram for grayscale image
hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])

#Plotting histogram with pyplot
plt.plot(hist)
plt.show()

#Binarization through thresholding
ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
#______________________________________________Test
cv2.imshow('Binary', img_bin)

#______________________________________________
#______________________________________________
#______________________________________________
#______________________________________________
#______________________________________________
#BLOB Detector starts
inputpic = img_gray
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

params.thresholdStep = 1
params.minThreshold = 0
params.maxThreshold = 255
params.minRepeatability
params.minDistBetweenBlobs = 0

params.filterByColor = 0
params.blobColor = 255

# Filter by Area.
params.filterByArea = 1
params.minArea = inputpic.shape[0]*inputpic.shape[1]*0
params.maxArea = inputpic.shape[0]*inputpic.shape[1]*1

params.filterByCircularity = 0
params.minCircularity = 0.1

params.filterByInertia = 0

params.filterByConvexity = 0

#______________________________________________
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
keypoints = detector.detect(inputpic)
#______________________________________________

# Draw detected blobs as red circles.
#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

#Empty numPy array
blank = np.zeros((1,1))

im_with_keypoints = cv2.drawKeypoints(img_bin, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#np.array([])

# Show blobs
cv2.imshow('Blob detection', im_with_keypoints)


#_______________AFTER EVERYTHING WAS DONE... IT ALL HAD TO BE DESTROYED..._____________
cv2.waitKey(0)
cv2.destroyAllWindows()