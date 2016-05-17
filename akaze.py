# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:54:50 2016

@author: firearasi
"""
#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt 

  #%%
cv2.ocl.setUseOpenCL(False)
img = cv2.imread('data/starbucks.jpg')
img1=cv2.imread('data/starbucks1.jpg')

# Initiate STAR detector
fd = cv2.AKAZE_create()
kp,des=fd.detectAndCompute(img,None)
kp1,des1=fd.detectAndCompute(img1,None)

matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
#FLANN_INDEX_LSH=6
#flann_params= dict(algorithm = FLANN_INDEX_LSH,
#                   table_number = 6, # 12
#                  key_size = 12,     # 20
#                   multi_probe_level = 1) #2
#matcher=cv2.FlannBasedMatcher(flann_params, {}) 
matches=matcher.match(des,des1)

matches = sorted(matches, key = lambda x:x.distance)


img2 = cv2.drawMatches(img,kp,img1,kp1,matches[:20],None, flags=2)

try:
  cv2.namedWindow("Pic",cv2.WINDOW_NORMAL)
  cv2.imshow("Pic",img2)
  cv2.resizeWindow("Pic",1920,1080)

  if cv2.waitKey(0)  ==ord('q'):
    cv2.destroyAllWindows()
except:
  pass
cv2.imwrite("output/starbucks_match.png",img2)
