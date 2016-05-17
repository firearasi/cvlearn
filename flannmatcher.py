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
img1 = cv2.imread('data/box.png')
img2=cv2.imread('data/box_in_scene.png')

# Initiate STAR detector
fd=cv2.ORB_create()
#fd = cv2.AKAZE_create()
kp1,des1=fd.detectAndCompute(img1,None)
kp2,des2=fd.detectAndCompute(img2,None)


FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
matcher=cv2.FlannBasedMatcher(flann_params,{}) 

matches=matcher.knnMatch(des1,des2,k=2)

#Need to draw only good matches, so create a mask


# ratio test as per Lowe's paper

def good_matches(matches,ratio=0.75):
  good_indices=[int(m.distance < ratio * n.distance) for (m,n) in matches]
  n=sum(good_indices)
  good_mask=[[i,0] for i in good_indices]
  return (n,good_mask)

r=0.75  
num_good_matches,matchesMask = good_matches(matches,ratio=r)
while num_good_matches>10:
    r -= 0.01
    num_good_matches,matchesMask = good_matches(matches,ratio=r)

print("Lowe ratio:",r)
print("Num of good matches:",num_good_matches)


draw_params = dict(matchColor = (0,0,255),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

cv2.imwrite("output/box_match.png",img3)



try:
  cv2.namedWindow("Pic",cv2.WINDOW_NORMAL)
  cv2.imshow("Pic",img3)
  cv2.resizeWindow("Pic",1920,1080)

  if cv2.waitKey(0)  ==ord('q'):
    cv2.destroyAllWindows()
except:
  pass

