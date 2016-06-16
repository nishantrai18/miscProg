import numpy as np
from skvideo.io import VideoCapture
import cv2
import skimage
import random

def equalizeImgList(imgList, row = 100, col = 100):
	'''
	Given a frameList, normalize (resize) all the frames to a common size
	'''
	# OPTIMIZATION POSSIBLE

	newFrameList = []

	for i in range(imgList.shape[0]):
		try:
			newImg = cv2.resize(imgList[i], (row, col))
		except:
			print 'Resizing error found for image with dimensions', imgList[i].shape
			continue

		newImg = np.array(newImg, dtype = np.uint8)
		newFrameList.append(newImg)

	newFrameList = np.array(newFrameList)
	return newFrameList

def getTruthVal(fileName):
	'''
	Take the fileName (CSV File) of the truth file
	Returns a dictionary which contains mapping between fileName and 5 attitude scores
	'''

	trueMap = {}
	trueVal = np.genfromtxt(fileName, delimiter=',', dtype=None)
	# Assuming we get a csv file

	for i in range(1,trueVal.shape[0]):
		# The first one is not a valid entry
		tmpList = list(trueVal[i][1:])
		tmpList = [float(x) for x in tmpList]
		tmpKey = trueVal[i][0].strip('.mp4')
		trueMap[tmpKey] = np.array(tmpList)

	return trueMap

def readFromFile(fileName, skipLength = 2, augment = False):
	filePath = 'tmpData/visualFetA/'
	fileName = filePath+fileName+'.npy'
	newImgList = np.load(fileName)
	tmpList = []
	row, col = 50, 50
	startInd = random.randint(0, skipLength-1)
	for i in range(startInd, newImgList.shape[0], skipLength):
		newImg = cv2.resize(newImgList[i], (row, col))
		if augment:
			if (random.randint(1,2) > 1):
				newImg = np.fliplr(newImg)
		tmpList.append(newImg)
	tmpList = np.array(tmpList)
	return tmpList

def readData(fileNames, trueVal = None):
	X = []
	Y = []
	# CAN BE OPTIMIZED
	for fileName in fileNames:
		imgList = readFromFile(fileName, 6, augment = True)
		X.extend(imgList)
		if (trueVal is not None):
			Y.extend([trueVal[fileName]]*len(imgList))
	X = np.array(X)
	Y = np.array(Y)
	return X, Y
