import cPickle as pickle
import numpy as np
import random
from readVideo import *

np.set_printoptions(precision=4)

def evaluateEnsemble(pred, trueVal, weights):
	p = np.dot(pred.transpose(), weights.transpose())

	errors = np.abs(p - trueVal)
	meanAccs = 1 - np.mean(errors, axis=0)
	
	return meanAccs

def getBestWeights(pred, trueVal, numModels, numIters):
	weightsMap = dict()

	weights = np.random.dirichlet(np.ones(numModels), size = numIters)

	for i in xrange(len(weights)):
		score = evaluateEnsemble(pred, trueVal, weights[i])
		weightsMap[i] = score
		# weightsMap[i] contains the score for weights[i]

	bestWeights = sorted(weightsMap, key = weightsMap.get, reverse = True)[:100]
	# print bestWeights
	# print weightsMap

	performers = np.array(sorted(weightsMap.values(), reverse = True)[:100])

	# print performers
	# for i in range(performers.shape[0]):
	# 	print weights[bestWeights[i]], weightsMap[bestWeights[i]]

	return weights, bestWeights

def getBestWeightsAfterVal(pred, trueVal, wMap, bWgt):
	print 'The best ensembles on validation'

	weightsMap = dict()

	for i in xrange(len(bWgt)):
		score = evaluateEnsemble(pred, trueVal, wMap[bWgt[i]])
		weightsMap[i] = score
		# weightsMap[i] contains the score for weights[i]

	bestWeights = sorted(weightsMap, key = weightsMap.get, reverse = True)[:20]

	performers = np.array(sorted(weightsMap.values(), reverse = True)[:20])
	print performers

	for i in range(performers.shape[0]):
		print wMap[bestWeights[i]], weightsMap[bestWeights[i]]

	return weightsMap, bestWeights

def getPrediction(modelName, vidNames):
	trueVal = pickle.load(open(modelName, 'rb'))
	# The object returned by pickle load should be a dictionary indexed by video name
	newList = []
	defVal = np.array([0.5]*5)
	for fileName in vidNames:
		if (fileName in trueVal):
			newList.append(trueVal[fileName])
		# else:
		# 	newList.append(defVal)

	newList = np.array(newList)
	# Should return a numpy array
	print newList.shape
	return newList

def convertToArray(trueVal, vidNames):
	newList = []
	defVal = np.array([0.5]*5)
	for fileName in vidNames:
		if (fileName in trueVal):
			newList.append(trueVal[fileName])
		# else:
		# 	newList.append(defVal)

	newList = np.array(newList)
	print 'HERE', newList.shape
	return newList

def getCommon(modelNames, vidNames):
	# Gets the common files present in all the model predictions

	newNames = []
	trueValList = []
	for i in range(len(modelNames)):
		trueValList.append(pickle.load(open(modelNames[i], 'rb')))

	for fileName in vidNames:
		flag = True
		for i in range(len(modelNames)):
			if fileName not in trueValList[i]:
				flag = False
				break
		if flag:
			newNames.append(fileName)

	return newNames

def createEnsemble(modelNames, numModels, numIters = 10000):
	videoPath = '../training/download_train-val/trainFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x.strip('.mp4') for x in vidNames if x.endswith(".mp4")]

	fileName = '../training/training_gt.csv'
	origTrueVal = getTruthVal(fileName)

	vidNames = getCommon(modelNames, vidNames)

	splitVal = 0.8

	vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
	vidNames = vidNames[:int(splitVal*len(vidNames))]

	trueVal = convertToArray(origTrueVal, vidNames)
	predList = []
	for i in range(numModels):
		predList.append(getPrediction(modelNames[i], vidNames))

	trueValTest = convertToArray(origTrueVal, vidNamesTest)
	predListTest = []
	for i in range(numModels):
		predListTest.append(getPrediction(modelNames[i], vidNamesTest))

	weightsMap = []
	bestWeights = []

	for i in range(5):
		pred = []
		predTest = []
		for j in range(numModels):
			pred.append(predList[j][:,i])
			predTest.append(predListTest[j][:,i])
		pred = np.array(pred)
		predTest = np.array(predTest)

		wMap, bWgt = getBestWeights(pred, trueVal[:,i], numModels, numIters)
		wMap, bWgt = getBestWeightsAfterVal(predTest, trueValTest[:,i], wMap, bWgt)

		weightsMap.append(wMap)
		bestWeights.append(bWgt)

	print bestWeights

if __name__ == "__main__":

	modelNames = ['tmpData/ensemble/mergeScore_FetAudioA_LS.p', 'tmpData/ensemble/mergeScore_FetC_Ridge.p', 
						'tmpData/ensemble/mergeScore_FetAudioA_WGT.p', 'tmpData/ensemble/mergeScore_FetC_WGT.p']

	numModels, numIters = len(modelNames), 10000
	createEnsemble(modelNames, numModels, numIters)	
