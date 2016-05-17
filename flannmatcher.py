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
img = cv2.imread('data/box.png')
img1=cv2.imread('data/box_in_scene.png')

# Initiate STAR detector
fd=cv2.ORB_create()
#fd = cv2.AKAZE_create()
kp,des=fd.detectAndCompute(img,None)
kp1,des1=fd.detectAndCompute(img1,None)

#matcher=cv2.BFMatcher(cv2.NORM_HAMMING2,crossCheck=True)
#matcher=cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)


FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
matcher=cv2.FlannBasedMatcher(flann_params,{}) 

matches=matcher.knnMatch(des,des1,k=2)

#Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,0,255),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img2 = cv2.drawMatchesKnn(img,kp,img1,kp1,matches,None,**draw_params)




try:
  cv2.namedWindow("Pic",cv2.WINDOW_NORMAL)
  cv2.imshow("Pic",img2)
  cv2.resizeWindow("Pic",1920,1080)

  if cv2.waitKey(0)  ==ord('q'):
    cv2.destroyAllWindows()
except:
  pass

cv2.imwrite("output/box_match.png",img2)
