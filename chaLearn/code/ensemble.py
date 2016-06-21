import cPickle as pickle
import numpy as np
import random
from readVideo import *

def evaluateEnsemble(pred, trueVal, weights):
	p = np.dot(pred, weights.transpose())

	errors = np.abs(p - trueVal)
	meanAccs = 1 - np.mean(errors, axis=0)
	
	return meanAccs

def getBestWeights(pred, trueVal, numModels, numIters):
	weightsMap = dict()

	weights = np.random.dirichlet(np.ones(numModels), size = numIters)

	for i in xrange(len(weights)):
		score = evaluateEnsemble(pred, trueVal, weights[i])
		weightsMap[weights[i]] = score

	bestWeights = sorted(weightsMap, key = weightsMap.get, reverse = True)[:100]
	print bestWeights
	print weightsMap

	return weightsMap, bestWeights

def getPrediction(modelName, vidNames):
	trueVal = pickle.load(open(modelName, 'rb'))
	# The object returned by pickle load should be a dictionary indexed by video name
	newList = []
	for fileName in vidNames:
		newList.append(trueVal[fileName])
	newList = np.array(newList)
	# Should return a numpy array
	return newList

def convertToArray(trueVal, vidNames):
	newList = []
	for fileName in vidNames:
		newList.append(trueVal[fileName])
	newList = np.array(newList)
	return newList

def createEnsemble(modelNames, numModels, numIters = 10000):
	videoPath = '../training/download_train-val/trainFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]

	fileName = '../training/training_gt.csv'
	trueVal = getTruthVal(fileName)
	trueVal = convertToArray(trueVal, vidNames)

	predList = []
	for i in range(numModels):
		predList.append(getPrediction(modelNames[i], vidNames))

	weightsMap = []
	bestWeights = []

	for i in range(5):
		pred = []
		for j in range(numModels):
			pred.append(predList[j][:,i])
		pred = np.array(pred)

		wMap, bWgt = getBestWeights(pred, trueVal[:,i], numModels, numIters)

		weightsMap.append(wMap)
		bestWeights.append(bWgt)

	print bestWeights

if __name__ == "__main__":

	modelNames = []
	numModels, numIters = len(modelNames), 1000
	createEnsemble(modelNames, numModels, numIters)	
