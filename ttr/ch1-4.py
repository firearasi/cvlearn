import cv2
import numpy as np
#print([x for x in dir(cv2) if x.startswith('COLOR_')])
img=cv2.imread('../data/apple.jpg')
h,w=img.shape[:2]
print((w,h))

rotation_matrix=cv2.getRotationMatrix2D((w/2,h/2),30,1)
img_rotation=cv2.warpAffine(img,rotation_matrix,(w,h))

cv2.imshow('Image',img)
cv2.imshow('rotated',img_rotation)
cv2.waitKey()
