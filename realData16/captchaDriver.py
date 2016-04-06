import os, glob
from captchaFunc import *

def readAllCaptchas():
	os.chdir("captchaTrainingSet/")
	X = []
	Y = []
	fileFormat = "inputTraining/"+"input*txt"
	inFiles = set(glob.glob(fileFormat))
	fileFormat = "outputTraining/"+"output*txt"
	outFiles = set(glob.glob(fileFormat))
	for fileName in inFiles:
		X.append(polishImg(fileName))
	for fileName in outFiles:
		Y.append(textRead(fileName))
	return X, Y

def main():
	X, Y = readAllCaptchas()

	for i in range(5):
		print 'THIS entry in X contains', len(X[i])
		for j in range(len(X[i])):
			if (j >= 5):
				break
			print Y[i][j]
			skimage.io.imshow(X[i][j])
			skimage.io.show()

main()