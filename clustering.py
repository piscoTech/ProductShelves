import numpy as np
import math

def naiveClustering(array, maxDist):
	clusters = []

	for y in range(array.shape[0]):
		for x in range(array.shape[1]):
			if array[y,x] != 255:
				continue

			possible = []
			for clId, cl in enumerate(clusters):
				for pt in cl:
					dist = math.sqrt((x-pt[0])**2 + (y-pt[1])**2)
					if dist <= maxDist:
						possible.append(clId)
						break

			if len(possible) == 0:
				clusters.append([(x,y)]) 
			else:
				# Combine clusters
				for cl in possible[1:][::-1]:
					clusters[possible[0]] += clusters[cl]
					del clusters[cl]

				clusters[possible[0]].append((x,y))

	return clusters

if __name__ == "__main__":
	a = np.zeros((9,9),dtype=np.uint8)
	a[1,2] = 255
	a[1,6] = 255
	a[3,4] = 255
	a[7,2] = 255
	print(a)

	cl = naiveClustering(a, 3)
	print("%d cluster(s)" % len(cl))
	for c in cl:
		print(c)
