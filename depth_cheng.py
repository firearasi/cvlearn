# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:54:50 2016

@author: firearasi
"""
#%%
from matching import *

args=get_args(pic_names_required=False)

name_left=args.img_left if args.img_left is not None else 'data/chengleft.png'
name_right=args.img_right if args.img_right is not None else 'data/chengright.png'

img_left = cv2.imread(name_left,0)
img_right=cv2.imread(name_right,0)

feature_type=args.feature_type
matcher_type=args.matcher_type

kps_left,des_left=get_descriptors(img_left,feature_type)
kps_right,des_right=get_descriptors(img_right,feature_type)

matches=get_matches(des_left,des_right,matcher_type)

n,matchesMask=get_n_good_matches_mask(matches,20)


draw_params = dict(matchColor = (0,0,255),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 2)

img = cv2.drawMatchesKnn(img_left,kps_left,
                                             img_right,kps_right,
                                            matches,None,**draw_params)

cv2.imwrite("output/indian_cheng.png",img)



try:
  cv2.namedWindow("Pic",cv2.WINDOW_NORMAL)
  cv2.imshow("Pic",img3)
  cv2.resizeWindow("Pic",1920,1080)

  if cv2.waitKey(0)  ==ord('q'):
    cv2.destroyAllWindows()
except:
  pass

