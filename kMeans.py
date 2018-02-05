import numpy
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance
from copy import deepcopy

filename1 = 'normal.txt'
filename2 = 'unbalanced.txt'

def readData(filename):
	xc = []
	yc = []
	coords = [xc, yc]
	with open(filename,'r') as f:
		for line in f:
			x, y = line.split()
			xc.append(float(x))
			yc.append(float(y))
	return coords

def chooseCentroids(k):
	centroidX = []
	centroidY = []
	index = numpy.random.randint(0, len(data[0]), size = k)
	centroids = [centroidX, centroidY]
	for i in range(0, k):
		centroidX.append(data[0][index[i]])
		centroidY.append(data[1][index[i]])
	return centroids

def computeError(current, previous):
	errorSum = 0
	for i in range(len(current)):
		point0 = [current[0][i], current[1][i]]
		point1 = [previous[0][i], previous[1][i]]
		error = distance.euclidean(point0, point1)
		errorSum = errorSum + error
	return errorSum

def kMeans(data, k):
#Step 1 - Picking K random points as cluster centers,i.e centroids.

	#centroids = [[x1,..,xk], [y1,..yk]], xi,yi - centroids coordinates
	centroids = [map(float, coord) for coord in chooseCentroids(k)]
	#plt.scatter(data[0], data[1])
	#plt.scatter(centroids[0], centroids[1] , c='r')
	#plt.show()

	#prevCentroids will be used for saving centroids coordinates before 
	#choosing another ones
	x = [0]
	y = [0]
	for i in range(len(centroids[0]) - 1):
		x.append(0)
		y.append(0)
	prevCentroids = [x, y]

	centroidNumbers = []

	error = computeError(centroids, prevCentroids)
	while error != 0:
#Step 2 - Assigning each point to nearest cluster by calculating its distance to 
#each centroid.
		centrN = 0
		#data[0] = [x1,...,xn]
		#data[1] = [y1,...,yn]
		#for each point [x, y] in data - compute Euclidean distance between
		#the point and each of centroid points. Then for each point find
		#to which centroid the distance is minimal.
		#In centroidNumbers each element represents the number of the centroid
		#to which corresponding point is closest, i.e:
		#centroidNumbers[c0=1, c1=0,..., cn=2] means that first point in data
		#is closest to centroid number 1, second point in data is closest to
		# centroid number 0 and so on.
		for pointN in range(len(data[0])):
			point = [data[0][pointN], data[1][pointN]]
			centroid = [centroids[0][0], centroids[1][0]]
			minDist = distance.euclidean(point, centroid)
			for i in range(1, k):
				centroid = [centroids[0][i], centroids[1][i]]
				currDist = distance.euclidean(point, centroid)
				if minDist > currDist:
					minDist = currDist
					centrN = i
			centroidNumbers.append(centrN)
			centrN = 0

			#copying old centroid coordinates in prevCentroids
			prevCentroids = deepcopy(centroids)

#Step 3 - Finding new centroids by taking the average of the assigned points.
		x = []
		y = []
		#cluster = [[x1,...,xn], [y1,...,yn]]
		cluster = [x, y]
		               #points in cluster0    #points in cluster1
		#clusters = [[[xk0,..], [yk0,...]], [[xk1,...], [yk1,...]],...]
		clusters = []
		for clustN in range(0, k):
			for pointN in range(len(data[0])):
				if clustN == centroidNumbers[pointN]:
					x.append(data[0][pointN])
					y.append(data[1][pointN])
			centroids[0][clustN] = numpy.mean(x)
			centroids[1][clustN] = numpy.mean(y)
			clusters.append(cluster)
			x = []
			y = []
		error = computeError(centroids, prevCentroids)
		#Step 4 - Repeat Step 2 and 3.

	return centroids, clusters

colors = ['r', 'g', 'r']
#data = 2-dimensional array coords=[[x1,...,xn], [y1,...,yn]]
data = [map(float, coord) for coord in readData(filename1)]
#data = [map(float, coord) for coord in readData(filename2)]
cluster_centers, ac = kMeans(data, 3)

fig, ax = plt.subplots()
for i in range(3):
	ax.scatter(ac[i][0], ac[i][1], c=colors[i])
ax.scatter(cluster_centers[0],cluster_centers[1], s=100, c='black')
plt.plot(ax)





