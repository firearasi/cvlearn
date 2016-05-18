import cv2
import numpy as np

img=cv2.imread('../data/cupwater.jpg')
img=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
h,w=img.shape[:2]

#h as x, w as y



cv2.imshow("Original",img)


kernel_vert=np.float32([[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]])
kernel_hor=kernel_vert.T

output_vert=cv2.filter2D(img,-1,kernel_vert)
cv2.imshow("Custom filter vert",output_vert)

output_hor=cv2.filter2D(img,-1,kernel_hor)
cv2.imshow("Custom filter hor",output_hor)

cv2.waitKey()
