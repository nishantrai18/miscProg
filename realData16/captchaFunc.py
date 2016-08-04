import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
import skimage.io
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import spectral_clustering
import scipy.ndimage
import scipy

def imgResize(img, dim):
	# print len(img), len(img[0])
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

def imgRead(fileName, threshold = 200, alpha = 20, beta = 10):
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
			l = (0.21*r + 0.72*g + 0.07*b)
			if (l > threshold):
				bgRow.append(0)
				r, g, b = 255, 255, 255
			else:
				bgRow.append(1)
				r, g, b = beta*(r/beta), beta*(g/beta), beta*(b/beta)            
			row.append([b,g,r])
			if (bgRow[-1]):
				pixelList.append([r*(1.0), g*(1.0), b*(1.0), alphaX*i, alphaY*j])
		bgImg.append(bgRow)
		img.append(row)

	pixelList = np.array(pixelList)
	bgImg = np.array(bgImg)
	img = np.array(img)

	f.close()

	return pixelList, img, bgImg, rV, cV

def getImages(pixelList, bgImg, rV, cV, numC = 6, dim = 32):

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

	newImage = np.ndarray((rV,cV,3))

	cnt = 0
	for i in range(0,rV):
		for j in range(0,cV):
			if (bgImg[i][j]):
				newImage[i][j][0:3] = centers[y[cnt]][0:3]
				cnt += 1
			else:
				newImage[i][j][0:3] = [255, 255, 255]

	newImage = newImage.astype(np.uint8)

	skimage.io.imshow(newImage)
	skimage.io.show()

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
	# print img
	# skimage.io.imshow(np.array(img, dtype = np.uint8))
	# skimage.io.show()
	return getImages(pixelList, bgImg, r, c, 6, dim = dim)

def showCurr(img, point, r, c):
	i = point
	print 'point', point
	print 'rows', r, c
	print int(i[0] - (r/2)), int(i[0] + (r/2)), int(i[1] - (c/2)), int(i[1] + (c/2))
	tmp = (img[int(i[0] - (r/2)) : int(i[0] + (r/2)), int(i[1] - (c/2)) : int(i[1] + (c/2))])
	# print tmp
	skimage.io.imshow(tmp)
	skimage.io.show()

def shiftMean(img, point, r, c, tr, tc):
	ans = [0.0, 0.0]
	wgt = 0.0
	# showCurr(img, point, r, c)
	print 'BEFORE SHIFT', point, r, c
	for i in range(int(point[0] - (r/2)), int(point[0] + (r/2)) + 1):
		if (i < 0):
			continue
		if (i >= tr):
			continue	
		for j in range(int(point[1] - (c/2)), int(point[1] + (c/2)) + 1):
			if (j < 0):
				continue
			if (j >= tc):
				continue
			ans[0] += img[i][j]*i*(1.0)
			ans[1] += img[i][j]*j*(1.0)		
			wgt += img[i][j]*(1.0)
	print 'HERE A ARE', ans, wgt
	ans = np.array(ans)
	ans = ans/wgt
	print 'HERWE ARE', point, ans
	return ans

def plotImg(img, initPoints, r, c):
	ansList = []
	for i in initPoints:
		ansList.append(img[int(i[0] - (r/2)) : int(i[0] + (r/2)), int(i[1] - (c/2)) : int(i[1] + (c/2))])
		skimage.io.imshow(ansList[-1])
		skimage.io.show()

	return ansList

def segmentImages(img, bgImg, r, c):
	initPoints = []
	wr, wc = int((0.65)*r), int(0.2*c)
	print 'Ws are', wr, wc
	num = 5
	for i in range(num):
		initPoints.append([(r/2), (i*(c/num)) + (wc/2)])

	for t in range(10):
		print initPoints
		for i in range(num):
			initPoints[i] = shiftMean(bgImg, initPoints[i], wr, wc, r, c)

	print initPoints
	plotImg(img, initPoints, wr, wc)
	return None

def segmentImg(fileName, dim = 32):
	pixelList, img, bgImg, r, c = imgRead(fileName)

	skimage.io.imshow(bgImg)
	skimage.io.show()

	return segmentImages(img, bgImg, r, c)

