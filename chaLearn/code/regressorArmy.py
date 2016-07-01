import cPickle as pickle
import numpy as np
import os
import random
import sklearn
from sklearn.svm import SVR, LinearSVR
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from dataProcess import *

def getParamValList(minVal, ratio = 10, sepSize = 4, listSize = 20):
	'''
	minVal, maxVal : Starting value, End value
	ratio : The ratio to multiply
	sepSize : The number of arithmetic jumps before multiplying base by ratio
	listSize : Number of elements
	'''

	paramList = []
	baseList = []

	for i in range(sepSize):
		baseList.append(minVal * (1.0 + ((ratio * i * 1.0)/sepSize)))

	baseList = np.array(baseList)
	paramList.extend(list(baseList))

	for j in range((listSize/sepSize) + 1):
		baseList = baseList * ratio
		paramList.extend(list(baseList))

	return paramList[ : listSize]
	# Only return the first few parameters

def getBaggedSet(modelName, bagAppend, dataAppend, j, paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal):
	# In the bagging used here, we only use consecutive parts since
	# spatially close points are similar training sets

	regName = modelName + bagAppend + '_' + dataAppend + str(j) + '_' \
						+ paramAppend + str(param) + '_' + bagIDAppend + str(k)

	bagSetSize = int(bagVal * X_train.shape[0])
	startInd = random.randint(0, X_train.shape[0] - bagSetSize - 1)
	X_tr, Y_tr = X_train[startInd : startInd + bagSetSize], Y_train[startInd : startInd + bagSetSize]
	# Selecting only a segment of the dataset

	return regName, X_tr, Y_tr

def createRegressorArmy(numTargets, regList, dataList, fetChoice, numLimit = 20, numPerParam = 4, accThreshold = 0.88, bagVal = 0.5, scaleFlag = False):
	'''
	Input: numTargets, The number of variables to predict
		   regList, List of possible regressor types (List of strings)
		   dataList, List of (Training, Testing) data pairs
					 (0.5, 0.5 split recommended)
		   fetChoice, The choice of feature selected
		   numLimit, Number limit for each type of classifier
		   numPerParam, Number of regressors trained per hyperparameter
		   bagVal, Bagging threshold for training multiple regressors
		   accThreshold, Accuracy/Score Threshold for models
		   scaleFlag, Scaling/Not Scaling flag, adds variety to the createClassifierArmy
	
	Output: List containing history of training

	Side effects: Write output to a file (folder in ensemble)
				  (Optional) Create list of dictionaries with the
				  predictions for another set (Ensure that the 
				  required data is given in the same order as dataList)
	'''

	'''
	Steps Involved:

		Consists of a set of classifiers already present
		Go through a loop to check if it is present in the provided arguments
		The inside statements consists of case statements
		Vary hyperparameters smoothly (mostly hardcoded), check and break on the basis of accThreshold
		For each hyperparameter, train numPerParam regressors via bagging (Around 0.75 (bagging Parameter)).
	'''

	availReg = ['SVR', 'BAG', 'Ridge', 'LS', 'LS+', 'RF', 'ADA', 'GBR']
	# Linear SVR, BaggingRegressor, Ridge, Lasso Regressor,
	# Lasso (With positive constraint), RandomForestRegressor,
	# AdaBoost Regressor, GradientBoostingRegressor

	savePath = 'tmpData/ensemble/audioFetA/' + fetChoice + '/'

	if not os.path.exists(savePath):
		os.makedirs(savePath)

	bagAppend = 'B_' + str(bagVal)
	# Refers to the bagging value used
	dataAppend = 'D_'
	# Refers to the data index
	bagIDAppend = 'ID_'
	# Maybe create a function for this

	histList = []

	for i in range(len(regList)):

		if (regList[i] not in availReg):
			continue

		for j in range(len(dataList)):

			X_train, Y_train = dataList[j][0][0], dataList[j][0][1]
			X_test, Y_test = dataList[j][1][0], dataList[j][1][1]

			if (regList[i] == 'SVR'):
				modelName = 'LinearSVR_'
				paramAppend = 'C_'
				# Gives the parameters for the regressor

				paramList = getParamValList(1e-5, 10, (numLimit/10), numLimit)

				for param in paramList:

					for k in range(numPerParam):

						regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, j, \
											  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)

						clfList = []
						Y_pred = np.zeros((X_test.shape[0], numTargets))					
						for i in range(numTargets):
							clfList.append(LinearSVR(C = param))
							clfList[i].fit(X_tr, Y_tr[:,i])
							Y_pred[:,i] = clfList[i].predict(X_test)

						Y_pred[Y_pred < 0] = 0
						Y_pred[Y_pred > 1] = 1

						score = evaluateTraits(Y_pred, Y_test, printFlag = False)
						histList.append([regName, score])

						if (score > accThreshold):
							pickle.dump(clfList, open(savePath + regName + '.p', 'wb'))						
						elif (accThreshold - score > (accThreshold/50)):
							break

			elif (regList[i] == 'BAG'):
				modelName = 'BAG_DecTree_'
				paramAppend = 'N_'
				# Gives the number of estimators

				paramList = getParamValList(10, 10, 4, 4)

				for param in paramList:

					param = int(param)
					for k in range(numPerParam):

						regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, j, \
											  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)

						clfList = []
						Y_pred = np.zeros((X_test.shape[0], numTargets))					
						for i in range(numTargets):
							clfList.append(BaggingRegressor(DecisionTreeRegressor(), n_estimators = param, n_jobs = 5))
							clfList[i].fit(X_tr, Y_tr[:,i])
							Y_pred[:,i] = clfList[i].predict(X_test)

						score = evaluateTraits(Y_pred, Y_test, printFlag = False)		
						histList.append([regName, score])

						if (score > accThreshold):
							pickle.dump(clfList, open(savePath + regName + '.p', 'wb'))
						elif (accThreshold - score > (accThreshold/100)):
							break

			elif (regList[i] == 'Ridge'):
				modelName = 'Ridge_'
				paramAppend = 'Alpha_'
				# Gives the parameters for the regressor

				paramList = getParamValList(1e-3, 10, (numLimit/10), numLimit)

				for param in paramList:

					for k in range(numPerParam):

						regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, j,
											  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)

						clfList = []
						Y_pred = np.zeros((X_test.shape[0], numTargets))					
						for i in range(numTargets):
							clfList.append(linear_model.Ridge(alpha = param))
							clfList[i].fit(X_tr, Y_tr[:,i])
							Y_pred[:,i] = clfList[i].predict(X_test)

						score = evaluateTraits(Y_pred, Y_test, printFlag = False)		
						histList.append([regName, score])

						if (score > accThreshold):
							pickle.dump(clfList, open(savePath + regName + '.p', 'wb'))
						elif (accThreshold - score > (accThreshold/100)):
							break

			elif (regList[i] == 'LS'):
				modelName = 'LS_'
				paramAppend = 'Alpha_'
				# Gives the parameters for the regressor

				paramList = getParamValList(1e-3, 10, (numLimit/10), numLimit)

				for param in paramList:

					for k in range(numPerParam):

						regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, j,
											  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)

						clfList = []
						Y_pred = np.zeros((X_test.shape[0], numTargets))					
						for i in range(numTargets):
							clfList.append(linear_model.Lasso(alpha = param))
							clfList[i].fit(X_tr, Y_tr[:,i])
							Y_pred[:,i] = clfList[i].predict(X_test)

						score = evaluateTraits(Y_pred, Y_test, printFlag = False)		
						histList.append([regName, score])

						if (score > accThreshold):
							pickle.dump(clfList, open(savePath + regName + '.p', 'wb'))
						elif (accThreshold - score > (accThreshold/100)):
							break

			elif (regList[i] == 'LS+'):
				modelName = 'Lasso+_'
				paramAppend = 'Alpha_'
				# Gives the parameters for the regressor

				paramList = getParamValList(1e-3, 10, (numLimit/10), numLimit)

				for param in paramList:

					for k in range(numPerParam):

						regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, j,
											  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)

						clfList = []
						Y_pred = np.zeros((X_test.shape[0], numTargets))					
						for i in range(numTargets):
							clfList.append(linear_model.Lasso(alpha = param, positive = True))
							clfList[i].fit(X_tr, Y_tr[:,i])
							Y_pred[:,i] = clfList[i].predict(X_test)

						score = evaluateTraits(Y_pred, Y_test, printFlag = False)		
						histList.append([regName, score])

						if (score > accThreshold):
							pickle.dump(clfList, open(savePath + regName + '.p', 'wb'))
						elif (accThreshold - score > (accThreshold/100)):
							break

			elif (regList[i] == 'RF'):
				modelName = 'RF_'
				paramAppend = 'N_'
				# Gives the parameters for the regressor

				paramList = getParamValList(10, 10, 4, 4)

				for param in paramList:

					param = int(param)
					for k in range(numPerParam):

						regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, j,
											  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)

						clfList = []
						Y_pred = np.zeros((X_test.shape[0], numTargets))					
						for i in range(numTargets):
							clfList.append(RandomForestRegressor(n_estimators = param, n_jobs = 5))
							clfList[i].fit(X_tr, Y_tr[:,i])
							Y_pred[:,i] = clfList[i].predict(X_test)

						score = evaluateTraits(Y_pred, Y_test, printFlag = False)		
						histList.append([regName, score])

						if (score > accThreshold):
							pickle.dump(clfList, open(savePath + regName + '.p', 'wb'))
						elif (accThreshold - score > (accThreshold/100)):
							break

			elif (regList[i] == 'ADA'):
				modelName = 'ADA_DecTree_'
				paramAppend = 'N_'
				# Gives the parameters for the regressor

				paramList = getParamValList(10, 10, 4, 4)

				for param in paramList:

					param = int(param)
					for k in range(numPerParam):

						regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, j,
											  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)

						clfList = []
						Y_pred = np.zeros((X_test.shape[0], numTargets))					
						for i in range(numTargets):
							clfList.append(AdaBoostRegressor(DecisionTreeRegressor(), n_estimators = param))
							clfList[i].fit(X_tr, Y_tr[:,i])
							Y_pred[:,i] = clfList[i].predict(X_test)

						score = evaluateTraits(Y_pred, Y_test, printFlag = False)		
						histList.append([regName, score])

						if (score > accThreshold):
							pickle.dump(clfList, open(savePath + regName + '.p', 'wb'))
						elif (accThreshold - score > (accThreshold/100)):
							break

			elif (regList[i] == 'GBR'):
				modelName = 'GBR_'
				paramAppend = 'N_'
				# Gives the parameters for the regressor

				paramList = getParamValList(10, 10, 4, 4)

				for param in paramList:

					param = int(param)
					for k in range(numPerParam):

						regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, j, \
												paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)

						clfList = []
						Y_pred = np.zeros((X_test.shape[0], numTargets))					
						for i in range(numTargets):
							clfList.append(GradientBoostingRegressor(n_estimators = param, loss='lad'))
							clfList[i].fit(X_tr, Y_tr[:,i])
							Y_pred[:,i] = clfList[i].predict(X_test)

						score = evaluateTraits(Y_pred, Y_test, printFlag = False)		
						histList.append([regName, score])

						if (score > accThreshold):
							pickle.dump(clfList, open(savePath + regName + '.p', 'wb'))
						elif (accThreshold - score > (accThreshold/100)):
							break

			else:
				print regList[i], "not implemented yet!"

			print i, j
			print (histList)

	print 'The history of training is as follows,'
	print histList

	return history

if __name__ == "__main__":
		
	np.set_printoptions(precision=2)

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

	regList = ['SVR', 'BAG', 'Ridge', 'LS', 'RF', 'ADA', 'GBR']

	fetChoice = 'AudioA_avg_ClusterFull'
	# X_train, Y_train = readData(vidNames, trueVal, feature = fetChoice, printFlag = True, clusterSize = -1)
	# print '\nReading training data complete'
	# X_test, Y_test = readData(vidNamesTest, trueVal, feature = fetChoice, printFlag = True, clusterSize = -1)
	# print '\nReading testing data complete'

	# dataList = [[[X_train, Y_train], [X_test, Y_test]]]
	# pickle.dump(dataList, open('tmpData/ensemble/audioFetA/audioFet_' + fetChoice + '.p', 'wb'))

	dataList = (pickle.load(open('tmpData/ensemble/audioFetA/audioFet_' + fetChoice + '.p', 'rb')))

	print 'Started creating regressor army', fetChoice

	createRegressorArmy(5, regList, dataList, fetChoice = 'Avg_ClusterFull', numLimit = 20, numPerParam = 5, accThreshold = 0.88, bagVal = 0.65)
