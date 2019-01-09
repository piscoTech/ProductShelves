import cv2 as cv
import numpy as np
import sys

def showImage(title, image, scale=1):
	image = cv.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
	cv.imshow(title, image)

def learnModel(productId):
	model = cv.imread('models/%d.jpg' % productId, -1)
	if model == None:
		print("Product %d not found!" % productId)
		sys.exit()

	kp, des = sift.detectAndCompute(model, None)	
	return (productId, kp, des, model.shape)

SCENES_CODE = "e"
SCENES_COUNT = 5
SCENES_EXT = 'png'
PRODUCTS = [0, 1, 27, 19, 24, 25, 26]

MIN_MATCH_COUNT = 30
FLANN_INDEX_KDTREE = 1
LOWE_RATIO_THRESHOLD = 0.4
MATCH_COLOR = (0,0,255)

print("Loading %d models..." % len(PRODUCTS))
sift = cv.xfeatures2d.SIFT_create()
models = map(learnModel, PRODUCTS)
print("Models loaded")

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)

for i in range(1, SCENES_COUNT + 1):
	print("\nAnalyzing scene %s%d..." % (SCENES_CODE, i))
	target = cv.imread('scenes/%s%d.%s' % (SCENES_CODE, i, SCENES_EXT), -1)
	if target == None:
		print("Scene %s%d not found!" % (SCENES_CODE, i))
		sys.exit()

	kpT, desT = sift.detectAndCompute(target, None)
	for p, kpM, desM, mShape in models:
		matches = flann.knnMatch(desM, desT, k = 2)
		# Store all the good matches as per Lowe's ratio test.
		good = [m for m, n in matches if m.distance < LOWE_RATIO_THRESHOLD * n.distance]

		if len(good) > MIN_MATCH_COUNT:
			print("Product %d found (%d matches)" % (p, len(good)))
			# Find homography
			src_pts = np.float32([kpM[m.queryIdx].pt for m in good]).reshape(-1,1,2)
			dst_pts = np.float32([kpT[m.trainIdx].pt for m in good]).reshape(-1,1,2)
			M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
			
			# Draw model bounding box in target
			h,w,d = mShape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			dst = cv.perspectiveTransform(pts,M)
			target = cv.polylines(target, [np.int32(dst)], True, MATCH_COLOR, 2, cv.LINE_AA)
		else:
			print("Product %d not found (%d matches)" % (p, len(good)))

	
	showImage("Scene %d" % i, target)
	if i < SCENES_COUNT:
		print("Press any key to move to the next scene (auto advancing in 10s)...")
		# cv.waitKey(10*1000)
		cv.waitKey()
	else:
		print("Press any key to exit...")
		cv.waitKey()

	cv.destroyAllWindows()
