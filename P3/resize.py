# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:08:37 2022

@author: Bruger
"""


import cv2

img=cv2.imread('pa.jpg')

imgW = img.shape[1]
imgH = img.shape[0]

skalar = 0.3

resized_dim = (int(imgW*skalar), int(imgH*skalar))


print(resized_dim)
resized = cv2.resize(img, resized_dim, interpolation=cv2.INTER_AREA)

cv2.imwrite('pa2.jpg', resized)

#cv2.imshow('resized',resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()