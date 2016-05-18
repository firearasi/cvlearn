import cv2
import numpy as np
#print([x for x in dir(cv2) if x.startswith('COLOR_')])
img=cv2.imread('../data/apple.jpg')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow('Image',img)
cv2.imshow('GrayImage',img_gray)
cv2.imshow('y ch',img_yuv[:,:,0])
cv2.imshow('u ch',img_yuv[:,:,1])
cv2.imshow('v ch',img_yuv[:,:,2])

cv2.waitKey()
