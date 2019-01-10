import cv2 as cv
import numpy as np
import sys

def showImage(title, image, scale=1):
	image = cv.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
	cv.imshow(title, image)

def learnModel(productId):
	model = cv.imread('models/%d.jpg' % productId, cv.IMREAD_COLOR)
	if model == None:
		print("Product %d not found!" % productId)
		sys.exit()

	kp, des = sift.detectAndCompute(model, None)	
	return (productId, kp, des, model.shape)

def drawKeypoints(scene, unused, used=[]):
	display = cv.drawKeypoints(scene, unused, None, KP_COLOR, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	display = cv.drawKeypoints(display, used, None, MATCH_COLOR, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	return display

SCENES_CODE = "e"
SCENES_COUNT = 5
SCENES_EXT = 'png'
PRODUCTS = [0, 27, 1, 19, 24, 25, 26]

MIN_MATCH_COUNT = 35
FLANN_INDEX_KDTREE = 1
LOWE_RATIO_THRESHOLD = 0.43
MATCH_COLOR = (0,0,255)
KP_COLOR = (0,255,0)

print("Loading %d models..." % len(PRODUCTS))
sift = cv.xfeatures2d.SIFT_create()
models = [learnModel(p) for p in PRODUCTS]
print("Models loaded")

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

for scnId in range(1, SCENES_COUNT + 1):
	print("\nAnalyzing scene %s%d..." % (SCENES_CODE, scnId))
	target = cv.imread('scenes/%s%d.%s' % (SCENES_CODE, scnId, SCENES_EXT), cv.IMREAD_COLOR)
	if target == None:
		print("Scene %s%d not found!" % (SCENES_CODE, scnId))
		sys.exit()

	kpT, desT = sift.detectAndCompute(target, None)
	usedKp = []
	showImage("Scene %d" % scnId, drawKeypoints(target, kpT))

	for p, kpM, desM, mShape in models:
		cv.waitKey()
		
		matches = flann.knnMatch(desM, desT, 2)
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

			# Filter used keypoints
			matchedKp = [m.trainIdx for m in good]
			usedKp += [kpT[m] for m in matchedKp]
			kpT = [kp for i,kp in enumerate(kpT) if i not in matchedKp]
			desT = np.array([d for i,d in enumerate(desT) if i not in matchedKp])
			showImage("Scene %d" % scnId, drawKeypoints(target, kpT, usedKp))
		else:
			print("Product %d not found (%d matches)" % (p, len(good)))
	
	if scnId < SCENES_COUNT:
		print("Press any key to move to the next scene (auto advancing in 10s)...")
	else:
		print("Press any key to exit...")
	
	cv.waitKey()
	cv.destroyAllWindows()
