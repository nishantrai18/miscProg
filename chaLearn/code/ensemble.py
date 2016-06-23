import cPickle as pickle
import numpy as np
import random
from readVideo import *
# from evaluateModel import *

np.set_printoptions(precision=4)

def evaluateEnsemble(pred, trueVal, weights):
	p = np.dot(pred.transpose(), weights.transpose())

	errors = np.abs(p - trueVal)
	meanAccs = 1 - np.mean(errors, axis=0)
	
	return meanAccs

def getRandomWeights(size, numSamples):
	weights = []
	for i in xrange(numSamples):
		weight = []
		for j in range(size):
			weight.append(random.randint(0,10))
		weight = np.array(weight)
		if (np.sum(weight) < 1):
			continue

		weight = (weight*(1.0))/np.sum(weight)
		weights.append(weight)
	weights = np.array(weights)
	return weights	

def getBestWeights(pred, trueVal, numModels, numIters):
	weightsMap = dict()

	# weights = np.random.dirichlet(np.ones(numModels), size = numIters)
	weights = getRandomWeights(numModels, numSamples = numIters)

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
		# print trueValList[i]

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

	splitVal = 0.9

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

def ensemblePredict(modelNames, weights, numModels):
	predList = []
	for i in range(numModels):
		predList.append(pickle.load(open(modelNames[i], 'rb')))

	predNew = {}
	for k in predList[0].keys():
		predNew[k] = []

	for k in predList[0].keys():
		for i in range(numModels):
			if (len(predList[i][k]) == 1):
				predList[i][k] = predList[i][k][0]

	for k in predList[0].keys():
		for i in range(5):
			predVal = 0
			sumTot = 0
			for j in range(numModels):
				predVal += weights[i][j]*predList[j][k][i]
				sumTot += weights[i][j]
			predVal /= sumTot
			predNew[k].append(predVal)

	for k in predList[0].keys():
		predNew[k] = np.array(predNew[k])

	generatePredFile(predNew)
	return predNew[k]

if __name__ == "__main__":

	# choice = 'create'
	choice = 'use'

	if (choice == 'create'):

		modelNames = ['tmpData/ensemble/mergeScore_FetAudioA_LS.p', 'tmpData/ensemble/mergeScore_FetC_Ridge.p', 
						'tmpData/ensemble/mergeScore_FetAudioA_WGT.p', 'tmpData/ensemble/mergeScore_FetC_WGT.p',
						'tmpData/ensemble/mergeScore_FetAudioA_BAG_LS.p']

		numModels, numIters = len(modelNames), 10000

		createEnsemble(modelNames, numModels, numIters)	

	else:

		# predNames = ['tmpData/predictions/valPredictionaudioFetA_MISC_LS.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_LS.p', 
		# 					'tmpData/predictions/valPredictionaudioFetA_MISC_WGT.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_WGT.p']

		predNames = ['tmpData/predictions/valPredictionaudioFetA_MISC_LS.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_LS.p', 
							'tmpData/predictions/valPredictionaudioFetA_MISC_WGT.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_WGT.p',
						'tmpData/predictions/valPredictionaudioFetA_BAG_n50_LS.p']


		# weights = [[0.376, 0.4, 0.098, 0.126], [0.1584, 0.432, 0.1466, 0.263], [0.0713, 0.5766, 0.1618, 0.1904], 
		# 				 [0.0905, 0.7361, 0.0786, 0.0947], [0.1688, 0.3957, 0.319, 0.1164]]

		# weights = [[0.376, 0.4, 0.098, 0.126], [0.1584, 0.432, 0.1466, 0.263], [0.1713, 0.5766, 0.1118, 0.1404], 
		# 				[0.0744, 0.4175, 0.5076, 0.0006], [0.1688, 0.3957, 0.319, 0.1164]]

		# weights = [[0.1046,0.1164, 0.1388, 0.2657, 0.0928, 0.2816], [0.0784, 0.1634, 0.0199, 0.352, 0.1712, 0.215], [0.2599, 0.0152, 0.0996, 0.0694, 0.3284, 0.2275],
		#  [0.1456, 0.1901, 0.2142, 0.149, 0.2226, 0.0784], [0.1928, 0.1221, 0.4763, 0.0196, 0.0228, 0.1664]]

		weights = [[0.0417, 0.3333, 0.375, 0.125, 0.125 ], [0.1667, 0.2778, 0.0, 0.1667, 0.3889], [0.2308, 0.3077, 0.0, 0.1154, 0.3462],
								 [0.25, 0.15, 0.2, 0.0, 0.4 ], [0.15, 0.35, 0.25, 0.05, 0.2]]

		weights = np.array(weights)
		ensemblePredict(predNames, weights, len(predNames))
