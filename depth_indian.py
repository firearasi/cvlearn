from matching import *
if __name__=='__main__':
		args=build_arg_parser().parse_args()
		img_left=cv2.imread(args.img_left,0)
		img_right=cv2.imread(args.img_right,0)
		feature_type=args.feature_type
		matcher_type=args.matcher_type
		print('feature type:',feature_type)
		print('matcher_type:',matcher_type)
		scaling_factor = 1.0
		img_left=cv2.resize(img_left, None,fx=scaling_factor,
																	fy=scaling_factor,
																	interpolation=cv2.INTER_AREA)
		img_right=cv2.resize(img_right, None,fx=scaling_factor,
																	fy=scaling_factor,
																	interpolation=cv2.INTER_AREA)
		kps_left,des_left=get_descriptors(img_left,feature_type)
		kps_right,des_right=get_descriptors(img_right,feature_type)
		
		matches=getMatches(des_left,des_right,matcher_type)
		
		
		
					
        





















