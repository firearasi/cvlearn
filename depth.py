#%%
import cv2
import cv2.xfeatures2d
import numpy as np
import matplotlib.pyplot as plt
cv2.ocl.setUseOpenCL(False)

#%%
img1=cv2.imread('data/left.jpg',0)
img2=cv2.imread('data/right.jpg',0)
sift=cv2.xfeatures2d.SIFT_create(nfeatures=200)

kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)
#%%
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann=cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.6*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1=np.int32(pts1)
pts2=np.int32(pts2)

#%% Find fundamental matrix
F,mask=cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
pts1=pts1[mask.ravel()==1]
pts2=pts2[mask.ravel()==1]

#%%
def drawlines(img1,img2,lines,pts1,pts2):
  ''' img1 - image on which we draw the epilines for the points in img2
      lines - corresponding epilines '''
  r,c = img1.shape
  img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
  img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
  for r,pt1,pt2 in zip(lines,pts1,pts2):
    color = tuple(np.random.randint(0,255,3).tolist())
    x0,y0 = map(int, [0, -r[2]/r[1] ])
    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
    img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
    img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
    img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
  return img1,img2
#%%
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.xticks([]),plt.yticks([])
plt.show()

#%%
def write_ply(fn, verts, colors):
  ply_header = '''ply
  format ascii 1.0
  element vertex %(vert_num)d
  property float x
  property float y
  property float z
  property uchar red
  property uchar green
  property uchar blue
  end_header
  '''
  verts = verts.reshape(-1, 3)
  colors = colors.reshape(-1, 3)
  verts = np.hstack([verts, colors])
  with open(fn, 'wb') as f:
    f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
    np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def write_obj(fn,verts):
  verts=verts.reshape(-1,3)
  with open(fn,'w') as f:
    for v in verts:
      f.write('v %f %f %f\n'%(v[0],v[1],v[2]))


imgL=cv2.imread('data/left.jpg')
imgR=cv2.imread('data/right.jpg')
window_size = 3
min_disp = 16
num_disp = 112-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 16,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
)

print('computing disparity...')
disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

print('generating 3d point cloud...',)
h, w = imgL.shape[:2]
f = 0.8*w                          # guess for focal length
Q = np.float32([[1, 0, 0, -0.5*w],
                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                [0, 0, 0,     -f], # so that y-axis looks up
                [0, 0, 1,      0]])
points = cv2.reprojectImageTo3D(disp, Q)
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
mask = disp > disp.min()
out_points = points[mask]
out_colors = colors[mask]
out_fn = 'out.ply'
write_ply('out.ply', out_points, out_colors)
print('%s saved' % 'out.ply')
write_obj('out.obj',out_points)

cv2.imshow('left', imgL)
cv2.imshow('disparity', (disp-min_disp)/num_disp)
cv2.waitKey(0)
cv2.destroyAllWindows()