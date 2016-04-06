import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
import skimage.io
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import spectral_clustering
import scipy.ndimage
import scipy

def imgResize(img, dim):
	print len(img), len(img[0])
	if (len(img) == 0)  or (len(img[0]) == 0):
		return img
	# tmpVal = ((len(img[0])*(1.0))/len(img))
	return scipy.misc.imresize(img, (dim, dim))

def getDim(imgList):
	if ((imgList is None) or (len(imgList) < 5)):
		return 0,0,0,0,0,0
	xList, yList = zip(*imgList)
	lx, hx = min(xList), max(xList)
	ly, hy = min(yList), max(yList)
	r = hx - lx + 1
	c = hy - ly + 1
	return lx, hx, ly, hy, r, c

def getImage(imgList):
	if ((imgList is None) or (len(imgList) < 5)):
		return None
	lx, hx, ly, hy, r, c = getDim(imgList)
	newImage = np.zeros((r,c))
	for p in imgList:
		newImage[p[0]-lx][p[1]-ly] = 1
	newImage.astype(np.uint8)
	return newImage

def imageSmooth(img):
	r = len(img)
	c = len(img[0])

	backImg = np.array(img)
	img = scipy.ndimage.filters.gaussian_filter(img, 2, mode='nearest')
	
	limit = 0.35
	newImg = np.array(img)
	newImg[newImg > limit] = 1
	newImg[newImg < limit] = 0

	tmpList = np.array(img).flatten()

	limit = 0.5
	coords = np.where(tmpList > limit)[0]
	imgList = [((i/c), i%c) for i in coords]

	lx, hx, ly, hy, nr, nc = getDim(imgList)
	img = scipy.ndimage.filters.gaussian_filter(backImg, 2, mode='nearest')[lx:hx,ly:hy]

	return img

def imgRead(fileName, threshold = 150, alpha = 1, beta = 10):
	f = open(fileName, "r")

	tmpStr = f.readline().split()
	rV, cV = int(tmpStr[0]), int(tmpStr[1])

	alphaX, alphaY = alpha*(255.0/rV), alpha*(255.0/cV)

	img = []
	bgImg = []
	pixelList = []
		
	for i in range(0,rV):
		line = f.readline()
		tmpLine = line.split()
		bgRow = []
		row = []
		for j in range(len(tmpLine)):
			x = tmpLine[j]
			val = x.split(',')
			r,g,b = int(val[0]), int(val[1]), int(val[2])
			l = (0.29*r + 0.59*g + 0.11*b)
			if (l > threshold):
				bgRow.append(0)
				r, g, b = 255, 255, 255
			else:
				bgRow.append(1)
				r, g, b = beta*(r/beta), beta*(g/beta), beta*(b/beta)            
			row.append([r,g,b])
			if (bgRow[-1]):
				pixelList.append([r*(1.0), g*(1.0), b*(1.0), alphaX*i, alphaY*j])
		bgImg.append(bgRow)
		img.append(row)

	pixelList = np.array(pixelList)
	bgImg = np.array(bgImg)
	img = np.array(img)

	f.close()

	return pixelList, img, bgImg, rV, cV

def getImages(pixelList, bgImg, rV, cV, numC = 7, dim = 32):

	limit = 10

	clf = KMeans(n_clusters = numC)

	y = clf.fit_predict(pixelList)

	centers = clf.cluster_centers_
	centers = centers.astype(np.uint8)
	centers = centers[centers[:,4].argsort()]

	cnt = 0
	imgList = [[] for i in range(numC)]

	for i in range(0,rV):
		for j in range(0,cV):
			if (bgImg[i][j]):
				imgList[y[cnt]].append((i,j))
				cnt += 1

	finalImg = np.ndarray((numC,dim,dim))
	for m in range(numC):
		tmpImg = getImage(imgList[m])
		tmpImg = imageSmooth(tmpImg)
		if not ((len(tmpImg) > limit) and (len(tmpImg[0]) > limit)):
			continue
		if ((len(tmpImg) != 0) and (len(tmpImg[0]) != 0) and (len(tmpImg[0]) < 50)):
			finalImg[m] = imgResize(tmpImg, dim)

	ans = []

	for m in range(numC):
		if (len(tmpImg) != 0):
			ans.append(finalImg[m]/255.0)

	return ans

def textRead(fileName):
	f = open(fileName, "r")
	letter = list(f.readline().strip('\n'))
	f.close()

	return letter

def polishImg(fileName, dim = 32):
	pixelList, img, bgImg, r, c = imgRead(fileName)

	return getImages(pixelList, bgImg, r, c, 7, dim = dim)