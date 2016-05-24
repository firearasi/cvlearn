# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:54:50 2016

@author: firearasi
"""
#%%
import cv2
import cv2.xfeatures2d
import numpy as np
import matplotlib.pyplot as plt 

#%%
cv2.ocl.setUseOpenCL(False)

import	argparse
def	get_args(pic_names_required = True):
	parser	=	argparse.ArgumentParser(description='Find	fundamental	matrix	\
									using	the	two	input	stereo	images	and	draw	epipolar	lines')
	parser.add_argument("-l","--img-left",	dest="img_left",	
																		required=pic_names_required,
																		help="Image	captured	from	the	left	view")
	parser.add_argument("-r","--img-right",	dest="img_right",	
																		required=pic_names_required,
																		help="Image	captured	from	the	right	view")
									
	parser.add_argument("-m","--matcher_type",
																		choices=['bf','flann'],default='bf',
																		help="keypoint matcher")
	parser.add_argument("-f","--feature-type",	dest="feature_type",
																		choices=['sift','surf','orb','akaze'] , default='akaze',
																		help="Feature	extractor	that	will	be	used;	can	be	either	'sift'	or	'surf'")
	
	return	parser.parse_args()
def	draw_lines(img_left,	img_right,	lines,	pts_left,	pts_right):				
	h,w	=	img_left.shapeimg_left	=	cv2.cvtColor(img_left,	cv2.COLOR_GRAY2BGR)
	img_right	=	cv2.cvtColor(img_right,	cv2.COLOR_GRAY2BGR)
	for	line,	pt_left,	pt_right	in	zip(lines,	pts_left,	pts_right):
		x_start,y_start	=	map(int,	[0,	-line[2]/line[1]	])
		x_end,y_end	=	map(int,	[w,	-(line[2]+line[0]*w)/line[1]	])
		color	=	tuple(np.random.randint(0,255,2).tolist())
		cv2.line(img_left,	(x_start,y_start),	(x_end,y_end),	color,1)
		cv2.circle(img_left,	tuple(pt_left),	5,	color,	-1)
		cv2.circle(img_right,	tuple(pt_right),	5,	color,	-1)
	return	img_left,	img_right

def	get_descriptors(gray_image,	feature_type):	
	if	feature_type	==	'surf':
		feature_extractor	=	cv2.xfeatures2d.SURF_create()
	elif	feature_type	==	'sift':
		feature_extractor	=	cv2.xfeatures2d.SIFT_create()
	elif feature_type == 'orb':
		feature_extactor = cv2.ORB_create()
	elif feature_type == 'akaze':
		feature_extractor = cv2.AKAZE_create()
	else:
		raise TypeError("The type of detector shoud be surf or sift!")
	return feature_extractor.detectAndCompute(gray_image,None)
	
def get_matches(des_left,des_right,matcher_type):
		if matcher_type == 'bf':
			matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
		elif matcher_type == 'flann':
				#	FLANN	parameters
				FLANN_INDEX_KDTREE	=	0
				index_params	=	dict(algorithm	=	FLANN_INDEX_KDTREE,	trees	=	5)
				search_params	=	dict(checks=50)
				#	Get	the	matches	based	on	the	descriptors
				matcher	=	cv2.FlannBasedMatcher(index_params,	search_params)
		else:
			raise TypeError('Wrong matcher type.')
		matches = matcher.knnMatch(des_left,des_right, k=2)
		return matches
def good_matches(matches,ratio=0.75):
  good_indices=[int(m.distance < ratio * n.distance) for (m,n) in matches]
  n=sum(good_indices)
  good_mask=[[i,0] for i in good_indices]
  return (n,good_mask)

def get_n_good_matches_mask(matches,n):	
	r=0.75  
	num_good_matches,matchesMask = good_matches(matches,ratio=r)
	while num_good_matches>n:
			r -= 0.01
			num_good_matches,matchesMask = good_matches(matches,ratio=r)
	return (num_good_matches,matchesMask)
	
