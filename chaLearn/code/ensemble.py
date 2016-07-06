import cPickle as pickle
import numpy as np
import random
from readVideo import *

import sklearn
from sklearn.svm import SVR, LinearSVR
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

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
			weight.append(random.randint(0,15))
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

	bestWeights = sorted(weightsMap, key = weightsMap.get, reverse = True)[:(numIters/100)]
	# print bestWeights
	# print weightsMap

	performers = np.array(sorted(weightsMap.values(), reverse = True)[:(numIters/100)])

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

	return wMap, weightsMap, bestWeights

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

def loadPredictions(modelNames, numModels, splitVal = 0.9):
	# Takes modelNames and numModels as input
	# Returns predList, predListTest, trueVal and trueValTest

	videoPath = '../training/download_train-val/trainFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x.strip('.mp4') for x in vidNames if x.endswith(".mp4")]

	fileName = '../training/training_gt.csv'
	origTrueVal = getTruthVal(fileName)

	vidNames = getCommon(modelNames, vidNames)

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

	return predList, predListTest, trueVal, trueValTest

def createEnsembleUsingLR(modelNames, numModels, choice = 'Ridge'):

	predList, predListTest, trueVal, trueValTest = loadPredictions(modelNames, numModels, splitVal = 0.9)

	X_train = predList[0]
	X_test = predListTest[0]
	for j in range(1, numModels):
		print X_train.shape, predList[j].shape
		X_train = np.concatenate((X_train, predList[j]), axis = 1)
		X_test = np.concatenate((X_test, predListTest[j]), axis = 1)

	print X_train.shape
	print X_test.shape

	Y_pred = np.zeros((X_test.shape[0], 5))
	
	clfList = []
	model_file_name = ''

	if (choice == 'Ridge'):
		model_file_name = 'tmpData/ensemble/ensemble' + str(numModels) + '_LR_Ridge'

		for i in range(5):
			print 'Currently training the', i, 'th regressor'
			clfList.append(linear_model.Ridge(alpha = 10))
			# clfList.append(linear_model.SGDRegressor())
			clfList[i].fit(X_train, trueVal[:,i])
			print 'Model Trained. Prediction in progress'
			Y_pred[:,i] = clfList[i].predict(X_test)

			print 'Predictions'
			print np.max(Y_pred[:,i])
			print np.min(Y_pred[:,i])
			print np.mean(Y_pred[:,i])
			print np.corrcoef(Y_pred[:,i], trueValTest[:,i])

			print 'Coefficents'
			print clfList[i].coef_
			print np.max(clfList[i].coef_)
			print np.min(clfList[i].coef_)
			print np.mean(clfList[i].coef_)

	elif (choice == 'SVR'):
		model_file_name = 'tmpData/ensemble/ensemble' + str(numModels) + '_LR_SVR'

		for i in range(5):
			print 'Currently training the', i, 'th regressor'
			clfList.append(SVR(C = 30.0, kernel = 'poly', degree = 2, coef0 = 1))

			clfList[i].fit(X_train, trueVal[:,i])
			print 'Model Trained. Prediction in progress'
			Y_pred[:,i] = clfList[i].predict(X_test)

			print 'Predictions'
			print np.max(Y_pred[:,i])
			print np.min(Y_pred[:,i])
			print np.mean(Y_pred[:,i])
			print np.corrcoef(Y_pred[:,i], trueValTest[:,i])

	elif (choice == 'BAG'):
		model_file_name = 'tmpData/ensemble/ensemble' + str(numModels) + '_LR_BAG'

		for i in range(5):
			print 'Currently training the', i, 'th regressor'
			# clfList.append(BaggingRegressor(DecisionTreeRegressor(), n_estimators = 50, n_jobs = 4))
			clfList.append(BaggingRegressor(SVR(C = 5), n_estimators = 50, n_jobs = 4))
			clfList[i].fit(X_train, trueVal[:,i])
			print 'Model Trained. Prediction in progress'
			Y_pred[:,i] = clfList[i].predict(X_test)

			print 'Predictions'
			print np.max(Y_pred[:,i])
			print np.min(Y_pred[:,i])
			print np.mean(Y_pred[:,i])
			print np.corrcoef(Y_pred[:,i], trueValTest[:,i])

	elif (choice == 'EXT'):
		model_file_name = 'tmpData/ensemble/ensemble' + str(numModels) + '_LR_EXT'

		for i in range(5):
			print 'Currently training the', i, 'th regressor'
			# clfList.append(BaggingRegressor(DecisionTreeRegressor(), n_estimators = 50, n_jobs = 4))
			clfList.append(ExtraTreesRegressor(n_estimators = 150, n_jobs = 4))
			clfList[i].fit(X_train, trueVal[:,i])
			print 'Model Trained. Prediction in progress'
			Y_pred[:,i] = clfList[i].predict(X_test)

			print 'Predictions'
			print np.max(Y_pred[:,i])
			print np.min(Y_pred[:,i])
			print np.mean(Y_pred[:,i])
			print np.corrcoef(Y_pred[:,i], trueValTest[:,i])



	# pickle.dump(clfList, open(model_file_name + '.p', 'wb'))

	print evaluateTraits(Y_pred, trueValTest)

def createEnsemble(modelNames, numModels, numIters = 10000):

	predList, predListTest, trueVal, trueValTest = loadPredictions(modelNames, numModels, splitVal = 0.92)

	weightsMap = []
	bestWeights = []
	bestEnsemble = []

	for i in range(5):
		pred = []
		predTest = []
		for j in range(numModels):
			pred.append(predList[j][:,i])
			predTest.append(predListTest[j][:,i])
		pred = np.array(pred)
		predTest = np.array(predTest)

		wMap, bWgt = getBestWeights(pred, trueVal[:,i], numModels, numIters)
		weights, wMap, bWgt = getBestWeightsAfterVal(predTest, trueValTest[:,i], wMap, bWgt)

		bestEnsemble.append(weights[bWgt[0]])
		weightsMap.append(wMap)
		bestWeights.append(bWgt)

	print "the best weight is"

	print '[',
	for i in range(len(bestEnsemble)):
		print '[',
		for j in range(len(bestEnsemble[i]) - 1):
			print bestEnsemble[i][j], ',',
		print bestEnsemble[i][-1],
		if (i < len(bestEnsemble) - 1):
			print '],'
	print ']'

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

def ensemblePredictUsingLR(modelNames, clfName, numModels):

	clfList = pickle.load(open(clfName + '.p', 'rb'))

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

	print 'HERE'

	cnt = 0

	for k in predList[0].keys():
		fetList = []
		for j in range(numModels):
			fetList.extend(predList[j][k])
		fetList = np.array(fetList)

		for i in range(5):
			predNew[k].append(clfList[i].predict([fetList])[0])
		print '\r', (cnt*(1.0))/len(predList[0]), 'part prediction completed',
		cnt += 1

	for k in predList[0].keys():
		predNew[k] = np.array(predNew[k])

	generatePredFile(predNew)
	return predNew[k]


if __name__ == "__main__":

	choice = 'create'
	# choice = 'use'
	# choice = 'useLR'

	if (choice == 'create'):

		modelNames = ['tmpData/ensemble/mergeScore_FetAudioA_LS.p', 'tmpData/ensemble/mergeScore_FetC_Ridge.p', 
						'tmpData/ensemble/mergeScore_FetAudioA_WGT.p', 'tmpData/ensemble/mergeScore_FetC_WGT.p',
						'tmpData/ensemble/mergeScore_FetAudioA_BAG_LS.p', 'tmpData/ensemble/mergeScore_FetAudioA_BAG_WGT.p',
						'tmpData/ensemble/mergeScore_FetC_48_96_LS.p']

		numModels, numIters = len(modelNames), 100000

		# createEnsemble(modelNames, numModels, numIters)	
		createEnsembleUsingLR(modelNames, numModels, choice = 'BAG')
		# EXT doesn't perform very well

	elif (choice == 'useLR'):

		predNames = ['tmpData/predictions/valPredictionaudioFetA_MISC_LS.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_LS.p', 
						'tmpData/predictions/valPredictionaudioFetA_MISC_WGT.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_WGT.p',
						'tmpData/predictions/valPredictionaudioFetA_BAG_n50_LS.p', 'tmpData/predictions/valPredictionaudioFetA_BAG_n50_WGT.p',
						'tmpData/predictions/valPredictionvisualFetC_Conv_48_96_256_LS.p']

		clfName = 'tmpData/ensemble/ensemble7_LR_BAG'

		ensemblePredictUsingLR(predNames, clfName, len(predNames))

	else:

		# predNames = ['tmpData/predictions/valPredictionaudioFetA_MISC_LS.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_LS.p', 
		# 					'tmpData/predictions/valPredictionaudioFetA_MISC_WGT.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_WGT.p']

		# predNames = ['tmpData/predictions/valPredictionaudioFetA_MISC_LS.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_LS.p', 
		# 					'tmpData/predictions/valPredictionaudioFetA_MISC_WGT.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_WGT.p',
		# 				'tmpData/predictions/valPredictionaudioFetA_BAG_n50_LS.p']

		predNames = ['tmpData/predictions/valPredictionaudioFetA_MISC_LS.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_LS.p', 
							'tmpData/predictions/valPredictionaudioFetA_MISC_WGT.p', 'tmpData/predictions/valPredictionvisualFetC_Conv_Augmented_32_64_256_WGT.p',
						'tmpData/predictions/valPredictionaudioFetA_BAG_n50_LS.p', 'tmpData/predictions/valPredictionaudioFetA_BAG_n50_WGT.p',
						'tmpData/predictions/valPredictionvisualFetC_Conv_48_96_256_LS.p']

		# weights = [[0.376, 0.4, 0.098, 0.126], [0.1584, 0.432, 0.1466, 0.263], [0.0713, 0.5766, 0.1618, 0.1904], 
		# 				 [0.0905, 0.7361, 0.0786, 0.0947], [0.1688, 0.3957, 0.319, 0.1164]]

		# weights = [[0.376, 0.4, 0.098, 0.126], [0.1584, 0.432, 0.1466, 0.263], [0.1713, 0.5766, 0.1118, 0.1404], 
		# 				[0.0744, 0.4175, 0.5076, 0.0006], [0.1688, 0.3957, 0.319, 0.1164]]

		# weights = [[0.1046,0.1164, 0.1388, 0.2657, 0.0928, 0.2816], [0.0784, 0.1634, 0.0199, 0.352, 0.1712, 0.215], [0.2599, 0.0152, 0.0996, 0.0694, 0.3284, 0.2275],
		#  [0.1456, 0.1901, 0.2142, 0.149, 0.2226, 0.0784], [0.1928, 0.1221, 0.4763, 0.0196, 0.0228, 0.1664]]

		# weights = [[0.0417, 0.3333, 0.375, 0.125, 0.125 ], [0.1667, 0.2778, 0.0, 0.1667, 0.3889], [0.2308, 0.3077, 0.0, 0.1154, 0.3462],
		# 						 [0.25, 0.15, 0.2, 0.0, 0.4 ], [0.15, 0.35, 0.25, 0.05, 0.2]]

		# weights = [[ 0.1982,0.6191,  0.1127,  0.0701, 0.1818, 0.0909], [0.1667, 0.2778, 0.0, 0.1429, 0.3889, 0.4286], [0.0713, 0.5766, 0.1618, 0.1904, 0.2, 0.0],
		# 						 [0.0526, 0.3158, 0.2105, 0.0526, 0.1053, 0.2632], [0.1688, 0.3957, 0.319, 0.1164, 0.2857, 0.2857]]

		weights = [ [ 0.0657894736842 , 0.184210526316 , 0.144736842105 , 0.171052631579 , 0.0921052631579 , 0.197368421053 , 0.144736842105 ],
					[ 0.0 , 0.152173913043 , 0.217391304348 , 0.152173913043 , 0.195652173913 , 0.0217391304348 , 0.260869565217 ],
					[ 0.150943396226 , 0.0188679245283 , 0.0943396226415 , 0.0754716981132 , 0.264150943396 , 0.283018867925 , 0.11320754717 ],
					[ 0.209302325581 , 0.116279069767 , 0.0 , 0.093023255814 , 0.348837209302 , 0.162790697674 , 0.0697674418605 ],
					[ 0.142857142857 , 0.0857142857143 , 0.228571428571 , 0.0285714285714 , 0.314285714286 , 0.0857142857143 , 0.114285714286 ] ]

		print weights

		weights = np.array(weights)
		ensemblePredict(predNames, weights, len(predNames))
