import cv2
import numpy as np

img=cv2.imread('../data/cupwater.jpg')
img=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
h,w=img.shape[:2]

#h as x, w as y

kernel_identity=np.array([[0,0,0],[0,1,0],[0,0,0]])
kernel_3x3=np.ones((3,3),np.float32)/9.0
kernel_5x5=np.ones((5,5),np.float32)/25.0

cv2.imshow("Original",img)

output1=cv2.filter2D(img,-1,kernel_identity)
cv2.imshow("Identity filter",output1)

output2=cv2.filter2D(img,-1,kernel_3x3)
cv2.imshow("3x3 filter",output2)

output3=cv2.filter2D(img,-1,kernel_5x5)
cv2.imshow("5x5 filter",output3)

kernel_custom=np.float32([[1,2,4,2,1],
                          [2,4,8,4,2],
                          [4,8,16,8,4],
                          [2,4,8,4,2],
                          [1,2,4,2,1]])
kernel_custom=kernel_custom/np.sum(kernel_custom)

output4=cv2.filter2D(img,-4,kernel_custom)
cv2.imshow("Custom filter",output4)

cv2.waitKey()
