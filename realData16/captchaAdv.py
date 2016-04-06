import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
import skimage.io
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import spectral_clustering
import scipy.ndimage

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
    # imgList is a list of tupels of the form
    # [X coord, Y coord]
    lx, hx, ly, hy, r, c = getDim(imgList)
    newImage = np.zeros((r,c))
    for p in imgList:
        newImage[p[0]-lx][p[1]-ly] = 1
    # newImage = imageSmooth(newImage, r, c)
    newImage.astype(np.uint8)
    return newImage

def imageSmooth(img):
    r = len(img)
    c = len(img[0])

    backImg = np.array(img)
    img = scipy.ndimage.filters.gaussian_filter(img, 2, mode='nearest')
    
    limit = 0.3
    newImg = np.array(img)
    newImg[newImg > limit] = 1
    newImg[newImg < limit] = 0

    skimage.io.imshow(newImg)
    skimage.io.show()

    tmpList = np.array(img).flatten()

    limit = 0.5
    coords = np.where(tmpList > limit)[0]
    imgList = [((i/c), i%c) for i in coords]

    lx, hx, ly, hy, nr, nc = getDim(imgList)

    img = newImg[lx:hx,ly:hy]

    print 'APNA'

    if (len(img) > 0):
        skimage.io.imshow(img)
        skimage.io.show()
    else:
        print 'NOT SHOWN'


    return img

tmpStr = raw_input().split()
rV, cV = int(tmpStr[0]), int(tmpStr[1])

threshold = 150

alpha = 0.5
beta = 10
alphaX, alphaY = alpha*(255.0/rV), alpha*(255.0/cV)

img = []
bgImg = []
pixelList = []
    
for i in range(0,rV):
    line = raw_input()
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

numC = 7

clf = KMeans(n_clusters = numC)

# bandwidth = estimate_bandwidth(pixelList, quantile=0.2, n_samples = 500)
# clf = MeanShift(bandwidth=bandwidth, bin_seeding=True)
y = clf.fit_predict(pixelList)

centers = clf.cluster_centers_

centers = centers.astype(np.uint8)

for i in range(len(centers)):
    centers[i][0:3] = np.random.randint(256, size=3)

print centers
print y, max(y), min(y)

cnt = 0
newImage = np.ndarray((numC,rV,cV,3))
imgList = [[] for i in range(numC)]

for i in range(0,rV):
    for j in range(0,cV):
        if (bgImg[i][j]):
            newImage[y[cnt]][i][j][0:3] = centers[y[cnt]][0:3]
            imgList[y[cnt]].append((i,j))
            cnt += 1
        else:
            for t in range(numC):
                newImage[t][i][j][0:3] = [255, 255, 255]

newImage = newImage.astype(np.uint8)

for m in range(numC):
    tmpImg = getImage(imgList[m])
    if (len(tmpImg) == 0):
        continue
    skimage.io.imshow(tmpImg)
    skimage.io.show()
    
    tmpImg = imageSmooth(tmpImg)
    if (len(tmpImg) == 0):
        continue

    skimage.io.imshow(tmpImg)
    skimage.io.show()


