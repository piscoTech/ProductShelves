import cv2 as cv
import numpy as np
import sys
from functools import reduce
from math import sqrt
import clustering as cl

def showImage(title, image, scale=1):
	image = cv.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
	cv.imshow(title, image)

def learnModel(productId):
	model = cv.imread('models/%d.jpg' % productId, cv.IMREAD_COLOR)
	if model == None:
		print("Product %d not found!" % productId)
		sys.exit()

	ref = (int(model.shape[1] / 2), int(model.shape[0] / 2))
	kp, des = sift.detectAndCompute(model, None)
	join = [np.array([[ref[0] - point.pt[0]], [ref[1] - point.pt[1]]]) for point in kp]
	return (productId, kp, des, model.shape, join)

def drawKeypoints(scene, unused, used=[]):
	display = cv.drawKeypoints(scene, unused, None, KP_COLOR, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	display = cv.drawKeypoints(display, used, None, MATCH_COLOR, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	return display

def naiveNSMThresh(img, threshold, size = 1):
	out = np.zeros_like(img, dtype=np.uint8)
	for y in range(size, img.shape[0] - size):
		for x in range(size, img.shape[1] - size):
			val = img[y,x]
			window = img[y-size:y+size+1, x-size:x+size+1]
			out[y,x] = 255 if val == np.max(window) and val >= threshold else 0

	return out

SCENES_CODE = "m"
SCENES_COUNT = 5
SCENES_EXT = 'png'
PRODUCTS = [0, 1, 11, 19, 24, 26, 25]

MIN_MATCH_COUNT = 18
MIN_MATCH_COUNT_SECONDARY = 15
FLANN_INDEX_KDTREE = 1
LOWE_RATIO_THRESHOLD = 0.4
MATCH_COLOR = (0,0,255)
KP_COLOR = (0,255,0)
AA_QUANTIZATION_STEP = 7
AA_THRESHOLD = 1
AA_NMS_SIZE = 2
AA_CLUSTERING_DISTANCE = 25

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
	tShape = target.shape
	foundCenters = []
	showImage("Scene %d" % scnId, drawKeypoints(target, kpT))

	for pId, kpM, desM, mShape, joinVec in models:
		cv.waitKey()
		
		matches = flann.knnMatch(desM, desT, 2)
		# Store all the good matches as per Lowe's ratio test.
		good = [m for m, n in matches if m.distance < LOWE_RATIO_THRESHOLD * n.distance]

		aa = np.zeros((tShape[0] / AA_QUANTIZATION_STEP, tShape[1] / AA_QUANTIZATION_STEP), dtype=np.uint8)
		aaMatches = {}
		aaShape = aa.shape
		for match in good:
			kpSrc = kpM[match.queryIdx]
			kpDst = kpT[match.trainIdx]
			join = joinVec[match.queryIdx]

			# model = np.ones(mShape)
			# model = cv.arrowedLine(model, (int(kpSrc.pt[0]), int(kpSrc.pt[1])), (int(kpSrc.pt[0] + join[0,0]), int(kpSrc.pt[1] + join[1,0])), (0,0,0), 5)
			# showImage("Model Scene", model)

			# tmp = np.ones(tShape)
			# tmp = cv.arrowedLine(tmp, (int(kpDst.pt[0]), int(kpDst.pt[1])), (int(kpDst.pt[0] + join[0,0]), int(kpDst.pt[1] + join[1,0])), (0,0,0), 1)
			# showImage("Target Scene", tmp)
			# cv.waitKey()

			# if (kpSrc.angle < 0) != (kpDst.angle < 0):
			# 	# Canonical orientation cannot be computed for only one KP
			# 	continue
			# elif kpSrc.angle >= 0:
			# 	theta = kpSrc.angle - kpDst.angle
			# 	rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
			# 						  [np.sin(theta),  np.cos(theta)]])
			# 	join = np.dot(rotMatrix, join)

			scale = kpSrc.size / kpDst.size
			join = scale * join
			hp = (kpDst.pt[0] + join[0][0], kpDst.pt[1] + join[1][0])

			# tmp = np.ones(tShape)
			# tmp = cv.arrowedLine(tmp, (int(kpDst.pt[0]), int(kpDst.pt[1])), (int(hp[0]), int(hp[1])), (0,0,0), 1)
			# showImage("Target Scene", tmp)
			# cv.waitKey()
			
			# Quantize the hypthesis with the same step as the AA
			hp = (int(hp[0] / AA_QUANTIZATION_STEP), int(hp[1] / AA_QUANTIZATION_STEP))
			if hp[0] < 0 or hp[0] >= aaShape[1] or hp[1] < 0 or hp[1] >= aaShape[0]:
				# The match gives an hypothesis outside the image, not a good match
				continue
			
			aa[hp[1], hp[0]] += 1
			if hp in aaMatches.keys():
				aaMatches[hp].append(match)
			else:
				aaMatches[hp] = [match]

		aa = naiveNSMThresh(aa, AA_THRESHOLD, size=AA_NMS_SIZE)
		showImage("AA", aa)
		voteClusters = cl.naiveClustering(aa, AA_CLUSTERING_DISTANCE)
		instances = []
		for instance in voteClusters:
			instanceMatches = []
			for pt in instance:
				instanceMatches += aaMatches[pt]

			if len(instanceMatches) < MIN_MATCH_COUNT:
				continue

			# Find homography(s)
			src_pts = np.float32([kpM[m.queryIdx].pt for m in instanceMatches]).reshape(-1,1,2)
			dst_pts = np.float32([kpT[m.trainIdx].pt for m in instanceMatches]).reshape(-1,1,2)
			while True:
				M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
				
				if M == None:
					break
				else:
					# Draw model bounding box in target
					h,w,d = mShape
					pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
					dst = cv.perspectiveTransform(pts,M)
					
					corners = [(c[0][0],c[0][1]) for c in dst]
					barycenter = reduce(lambda c1, c2: (c1[0]+c2[0], c1[1]+c2[1]), corners)
					barycenter = (barycenter[0]/4, barycenter[1]/4)
					target = cv.polylines(target, [np.int32(dst)], True, MATCH_COLOR, 2, cv.LINE_AA)

					width = (corners[3][0] + corners[2][0] - corners[1][0] - corners[0][0]) / 2
					height = (corners[2][1] + corners[1][1] - corners[0][1] - corners[3][1]) / 2

					instances.append((barycenter[0], barycenter[1], width, height, len(src_pts)))

					src_pts = np.float32([pt for i, pt in enumerate(src_pts) if not mask[i][0]])
					dst_pts = np.float32([pt for i, pt in enumerate(dst_pts) if not mask[i][0]])

					if len(src_pts) < MIN_MATCH_COUNT_SECONDARY:
						break

		if len(instances) == 0:
			print("Product %d not found" % pId)
		else:
			print("Product %d found, %d instance(s)" % (pId, len(instances)))
			for inst in instances:
				print("\tPosition (%d,%d), Width %dpx, Height %dpx, Matches %d" % inst)
			
			showImage("Scene %d" % scnId, target)
	
	if scnId < SCENES_COUNT:
		print("Press any key to move to the next scene...")
	else:
		print("Press any key to exit...")
	
	cv.waitKey()
	cv.destroyAllWindows()
