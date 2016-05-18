import cv2
import numpy as np
#print([x for x in dir(cv2) if x.startswith('COLOR_')])
img=cv2.imread('../data/apple.jpg')
h,w=img.shape[:2]
print((w,h))

src_points=np.float32([[0,0],[w-1,0],[0,h-1]])
dest_points=np.float32([[0,int(0.1*h)],[int(0.7*w),int(0.2*h)],[int(0.2*w),int(0.67*h)]])
affine_matrix=cv2.getAffineTransform(src_points,dest_points)
img_output=cv2.warpAffine(img,affine_matrix,(w,h))
cv2.imshow('Image',img)

cv2.imshow('Output',img_output)
cv2.waitKey()
