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

def getPrediction(modelName, vidNames, thresholdA = 0.88, thresholdB = 0.8915):
	tmpObj = pickle.load(open(modelName, 'rb'))
	stats, trueVal = tmpObj[0], tmpObj[1]

	if ((stats['accB'] < thresholdB) or (stats['accA'] < thresholdA)):
		return 0, None

	print stats
	# The object returned by pickle load should be a dictionary indexed by video name
	newList = []
	defVal = np.array([0.5]*5)
	for fileName in vidNames:
		if (fileName in trueVal):
			tmpList = trueVal[fileName]
			if ('numpy.float' not in str(type(tmpList))):
				newList.append(tmpList)
			else:
				newList.append([0.5]*5)

	newList = np.array(newList)
	# Should return a numpy array
	# print newList.shape
	return stats['accB'], newList

def convertToArray(trueVal, vidNames):
	newList = []
	defVal = np.array([0.5]*5)
	for fileName in vidNames:
		if (fileName in trueVal):
			newList.append(trueVal[fileName])

	newList = np.array(newList)
	# print 'HERE', newList.shape
	return newList

def loadPredictions(numTargets, modelPath, vidNames, splitVal = 0.9):
	# Returns predList, predListTest, trueVal and trueValTest

	modelNames = os.listdir(modelPath)
	modelNames = [(modelPath + x) for x in modelNames if x.startswith('predictDict_')]

	fileName = '../training/training_gt.csv'
	origTrueVal = getTruthVal(fileName)

	vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
	vidNames = vidNames[:int(splitVal*len(vidNames))]

	trueVal = convertToArray(origTrueVal, vidNames)
	predList = []
	accList = []
	for i in range(len(modelNames)):
		acc, tmpPred = getPrediction(modelNames[i], vidNames)
		if (tmpPred is not None):
			predList.append(tmpPred)
			accList.append(acc)

	trueValTest = convertToArray(origTrueVal, vidNamesTest)
	predListTest = []
	for i in range(len(modelNames)):
		_, tmpPred = getPrediction(modelNames[i], vidNamesTest)
		if (tmpPred is not None):
			predListTest.append(tmpPred)

	return accList, predList, predListTest, trueVal, trueValTest

def getCorrelation(predList, numID):
	# numID is the id of the trait for which correlation is to be computed
	X = predList[0][:,numID].reshape(predList[0].shape[0], 1)
	# print X.shape
	for j in range(1, len(predList)):
		# print X.shape, predList[j].shape
		X = np.concatenate((X, predList[j][:,numID].reshape(predList[j].shape[0], 1)), axis = 1)

	corr = np.corrcoef(X, rowvar = 0)
	print corr
	return corr

def getBestModel(corr, modelList, candList):
	optCand = 0
	minSum = len(modelList)
	for c in candList:
		suma = 0
		for i in modelList:
			suma += corr[c][i]
			# suma += 0
		if (suma < minSum):
			minSum = suma
			optCand = c
	return optCand

def correlation(numTargets, modelPath, vidNames, maxModels = 20, choice = 'Ridge'):

	accList, predList, predListTest, trueVal, trueValTest = loadPredictions(numTargets, modelPath, vidNames, splitVal = 0.5)

	corrList = []
	for i in range(numTargets):
		corrList.append(getCorrelation(predList, i))


def createEnsembleUsingLR(modelList, predList, predListTest, trueVal, trueValTest, choice = 'BAG'):

	X_train = predList[modelList[0]]
	X_test = predListTest[modelList[0]]
	for j in range(1, len(modelList)):
		# print X_train.shape, predList[modelList[j]].shape
		X_train = np.concatenate((X_train, predList[modelList[j]]), axis = 1)
		X_test = np.concatenate((X_test, predListTest[modelList[j]]), axis = 1)

	numModels = len(modelList)
	print numModels

	print X_train.shape
	print X_test.shape

	Y_pred = np.zeros((X_test.shape[0], 5))
	
	clfList = []
	model_file_name = ''

	if (choice == 'Ridge'):
		model_file_name = 'tmpData/ensemble/ensemble' + str(numModels) + '_LR_Ridge'

		for i in range(5):
			print 'Currently training the', i, 'th regressor'
			clfList.append(linear_model.Ridge(alpha = 2))
			# clfList.append(linear_model.SGDRegressor())
			clfList[i].fit(X_train, trueVal[:,i])
			print 'Model Trained. Prediction in progress'
			Y_pred[:,i] = clfList[i].predict(X_test)

			print 'Coefficents'
			print clfList[i].coef_
			print np.max(clfList[i].coef_)
			print np.min(clfList[i].coef_)
			print np.mean(clfList[i].coef_)

			print 'Predictions'
			print np.max(Y_pred[:,i])
			print np.min(Y_pred[:,i])
			print np.mean(Y_pred[:,i])
			print np.corrcoef(Y_pred[:,i], trueValTest[:,i])

	elif (choice == 'SVR'):
		model_file_name = 'tmpData/ensemble/ensemble' + str(numModels) + '_LR_SVR'

		for i in range(5):
			print 'Currently training the', i, 'th regressor'
			clfList.append(SVR(C = 1.0, kernel = 'poly', degree = 2, coef0 = 0.1))

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
			clfList.append(BaggingRegressor(SVR(C = 5, degree = 1), n_estimators = 50, n_jobs = 4))
			clfList[i].fit(X_train, trueVal[:,i])
			print 'Model Trained. Prediction in progress'
			Y_pred[:,i] = clfList[i].predict(X_test)
		
			print 'Predictions'
			print np.max(Y_pred[:,i])
			print np.min(Y_pred[:,i])
			print np.mean(Y_pred[:,i])
			print np.corrcoef(Y_pred[:,i], trueValTest[:,i])


	print evaluateTraits(Y_pred, trueValTest)


def createEnsemble(numTargets, modelPath, vidNames, maxModels = 20, choice = 'SVR'):

	accList, predList, predListTest, trueVal, trueValTest = loadPredictions(numTargets, modelPath, vidNames, splitVal = 0.5)

	corrList = []
	for i in range(numTargets):
		corrList.append(getCorrelation(predList, i))

	prefList = sorted(range(len(accList)), key=lambda k: accList[k], reverse = True)

	if (len(prefList) < maxModels):
		maxModels = len(prefList)

	modelList = []
	candList = []
	for j in range(numTargets):
		candList.append(list(prefList))
		tmpList = [prefList[0]]
		modelList.append(tmpList)
		candList[j].remove(tmpList[0])

	for j in range(numTargets):
		for i in range(maxModels):
			optCand = getBestModel(corrList[j], modelList[j], candList[j])
			modelList[j].append(optCand)
			candList[j].remove(optCand)

	modelSet = set()
	for i in range(numTargets):
		modelSet |= set(modelList[i])
	modelSet = list(modelSet)

	createEnsembleUsingLR(modelSet, predList, predListTest, trueVal, trueValTest)

	print modelList

if __name__ == "__main__":
		
	np.set_printoptions(precision=3)

	videoPath = '../training/download_train-val/trainFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]

	fileName = '../training/training_gt.csv'
	trueVal = getTruthVal(fileName)

	for i in xrange(len(vidNames)):
		vidNames[i] = vidNames[i].strip('.mp4')

	vidNames = vidNames
	splitVal = 0.5
	vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
	vidNames = vidNames[:int(splitVal*len(vidNames))]

	modelPath = 'tmpData/ensemble/audioFetAMultiStage/AudioA_avg_minmax_predictions/'

	# correlation(5, modelPath, vidNamesTest)
	createEnsemble(5, modelPath, vidNamesTest)