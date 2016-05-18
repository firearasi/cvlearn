import cv2
import numpy as np

img=cv2.imread('../data/cupwater.jpg')
img=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
h,w=img.shape[:2]

#h as x, w as y



cv2.imshow("Original",img)

sobel_horizontal=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
sobel_vertical=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)

cv2.imshow("Custom filter vert",sobel_vertical)


cv2.imshow("Custom filter hor",sobel_horizontal)

laplacian=cv2.Laplacian(gray,cv2.CV_64F)
cv2.imshow("Laplacian",laplacian)

canny=cv2.Canny(gray,180,300)
cv2.imshow("Canny",canny)

cv2.waitKey()
