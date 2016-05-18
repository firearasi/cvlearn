import cv2
import numpy as np
#print([x for x in dir(cv2) if x.startswith('COLOR_')])
img=cv2.imread('../data/apple.jpg')
h,w=img.shape[:2]
print((w,h))

img_scaled1=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
img_scaled2=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)


cv2.imshow('Image',img)
cv2.imshow('scaled1',img_scaled1)
cv2.imshow('scaled2',img_scaled2)
cv2.waitKey()
