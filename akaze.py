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
img1 = cv2.imread('data/starbucks.jpg')
img2=cv2.imread('data/starbucks1.jpg')

# Initiate STAR detector
fd = cv2.AKAZE_create()
kp1,des1=fd.detectAndCompute(img1,None)
kp2,des2=fd.detectAndCompute(img2,None)

#matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True) crossCheck = True only when k=1
matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

matches=matcher.knnMatch(des1,des2,k=2)

matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
num_good_matches = 0
for i,(m,n) in enumerate(matches):
    if m.distance < 0.6*n.distance:
        matchesMask[i]=[1,0]
        num_good_matches += 1

print("Num of good matches:",num_good_matches)


draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

try:
  cv2.namedWindow("Pic",cv2.WINDOW_NORMAL)
  cv2.imshow("Pic",img3)
  cv2.resizeWindow("Pic",1920,1080)

  if cv2.waitKey(0)  ==ord('q'):
    cv2.destroyAllWindows()
except:
  pass
cv2.imwrite("output/starbucks_match.png",img3)
