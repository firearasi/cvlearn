import cv2
import numpy as np
#print([x for x in dir(cv2) if x.startswith('COLOR_')])
img=cv2.imread('../data/apple.jpg')
r,c=img.shape[:2]
print((r,c))

translation_matrix=np.float32([ [1,0,70],[0,1,110] ])
img_translation=cv2.warpAffine(img,translation_matrix,(c,r))

cv2.imshow('Image',img)
cv2.imshow('Translated',img_translation)
cv2.waitKey()
