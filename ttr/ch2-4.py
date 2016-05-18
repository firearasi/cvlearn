import cv2
import numpy as np

img=cv2.imread('../data/tree.jpg')
#img=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

h,w=img.shape[:2]
#h as x, w as y

 
cv2.imshow("Original",img)

size=15
kernel_motion_blur=np.zeros((size,size),np.float32)
kernel_motion_blur[(size-1)//2,:]=np.ones(size)
kernel_motion_blur=kernel_motion_blur/np.sum(kernel_motion_blur)

kernel_motion_blue_vert=kernel_motion_blur.T

kernel_motion_blur_diag=np.zeros((size,size),np.float32)

for i in range(size):
  kernel_motion_blur_diag[i,i]=1  

for i,kernel in enumerate([kernel_motion_blur,kernel_motion_blue_vert,   kernel_motion_blur_diag]):
  output=cv2.filter2D(img,-1,kernel_motion_blur)
  cv2.imshow("Motion blur %d"%i,output)

cv2.waitKey()
