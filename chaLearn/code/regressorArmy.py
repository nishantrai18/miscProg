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
from sklearn.ensemble import AdaBoostRegressor

from dataProcess import *

class MultiStageRegressor():

	def __init__(self):
		self.regA = []
		# regA is the stage 1 regressor, which gives predictions for smaller instances
		self.regB = []
		# regB joins the predictions of the instances and predicts the final scores
		self.regBFetSize = 0
		self.accA = 0
		self.accB = 0
		self.name = ''

	def setName(self, name):
		# Set the name of the multiStageRegressor object
		self.name = name

	def trainStageA(self, reg, numTargets, X_train, Y_train):
		self.regA = []
		for i in range(numTargets):
			(self.regA).append(sklearn.base.clone(reg))
			(self.regA)[i].fit(X_train, Y_train[:,i])

	def getStageAAcc(self, numTargets, X_test, Y_test):
		Y_pred = np.zeros((X_test.shape[0], numTargets))
		for i in range(numTargets):
			Y_pred[:,i] = (self.regA)[i].predict(X_test)
		self.accA = evaluateTraits(Y_pred, Y_test, printFlag = False)
		print 'Accuracy of stage A is', self.accA

	def getFetStageB(self, numTargets, trainSet):
		# Returns the features for stage B

		X_train = []

		for i in range(len(trainSet)):
			Y_pred = np.zeros((trainSet[i].shape[0], numTargets))
			for j in range(numTargets):
				Y_pred[:,i] = (self.regA)[i].predict(trainSet[i])
			while (Y_pred.shape[0] < self.regBFetSize):
				Y_pred = np.concatenate((Y_pred, np.mean(Y_pred, axis = 0)), axis = 0)
			# Fixing the data in case there are less features than required	
			X_train.append(Y_pred.flatten())
		X_train = np.array(X_train)

		return X_train

	def trainStageB(self, reg, numTargets, trainSet, Y_train):
		# trainSet is a list of X_trains's, i.e. one for each sample

		self.regB = []
		# Identify the ideal size of features for the merging regressor
		maxLen = 0
		for i in range(len(trainSet)):
			maxLen = max(maxLen, trainSet[i].shape[0])
		self.regBFetSize = maxLen

		X_train = self.getFetStageB(numTargets, trainSet)

		for i in range(numTargets):
			(self.regB).append(sklearn.base.clone(reg))
			(self.regB)[i].fit(X_train, Y_train[:,i])

	def getStageBAcc(self, numTargets, testSet, Y_test):
		X_test = self.getFetStageB(numTargets, testSet)

		Y_pred = np.zeros((X_test.shape[0], numTargets))
		for i in range(numTargets):
			Y_pred[:,i] = (self.regB)[i].predict(X_test)

		self.accB = evaluateTraits(Y_pred, Y_test, printFlag = False)
		print 'Accuracy of stage B is', self.accB


def getMultiStageSet(vidNames, trueVal, fetChoice, clusterSize, printFlag = True):
	# Returns the train and test set, as required by MultiRegressorArmy

	multiSet = []
	Y = []
	X = []

	for i in range(len(vidNames)):
		tmpX, tmpY = readData([vidNames[i]], trueVal, feature = fetChoice, clusterSize = clusterSize)
		if (printFlag):
			print '\r', (i*(1.0))/len(vidNames), 'part reading completed',
			sys.stdout.flush()

		if (tmpX.shape[0] == 0):
			continue
		multiSet.append(tmpX)
		X.append(tmpX[0])
		Y.append(tmpY[0])

	Y = np.array(Y, dtype = np.float16)
	X = np.array(X, dtype = np.float16)

	return multiSet, X, Y

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

def createRegressorArmy(numTargets, regList, dataList, fetChoice, numLimit = 10, numPerParam = 4, accThreshold = 0.885, bagVal = 0.5, scaleFlag = False):
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

				paramList = getParamValList(1e-3, 10, (numLimit/5), numLimit)

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

				paramList = getParamValList(1e-3, 10, (numLimit/5), numLimit)

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

				paramList = getParamValList(1e-3, 10, (numLimit/5), numLimit)

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

				paramList = getParamValList(1e-3, 10, (numLimit/5), numLimit)

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

	return histList

def createAudioMultiStageRegressorArmy(numTargets, regList, vidNames, vidNamesTest, mergeList, fetList, segmentList, fetChoice, \
									   numLimit = 10, numPerParam = 4, accThreshold = 0.885, bagVal = 0.5, scaleFlag = False):
	'''
	Specific regressor factory for audio features, creates features with multiple segments
	Input: numTargets, The number of variables to predict
		   regList, List of possible regressor types (List of strings)
		   vidNames, names of video files to be used as training features
		   vidNamesTest, names of video files to be used as testing features
		   mergeList, List of possible merging regressors (Include LinearSVR, polySVR, LS+, etc)
		   fetList, List of types of data
		   segmentList, List of possible segments to try (Directly use this and fetList to save and create dataList)
		   fetChoice, The choice of features selected
		   numLimit, Number limit for each type of regressor
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
		
		Have a class of predictors which takes a list of features as input (opposed to just one feature set in normal classifiers) \
		and does a multi stage prediction on it. It contains of two regressors inside.
		Have a set of valid regressors
		(1, 2) Go through a loop through fetList and then through segmentList
		(3) Go through a loop to check if it is present in the provided arguments (regList)
		The inside statements consists of case statements
		(4) Vary hyperparameters smoothly (mostly hardcoded), check and break on the basis of accThreshold
		(5) For each hyperparameter, train numPerParam regressors via bagging (Around 0.65 (bagging Parameter)).
		After training on the bagged set, take the predictions of X_train (complete) and use it to train the merging regressor
		(6) Go through a loop through mergeList to check the type of regressor (Have only one of each type, no hyperparameter varying)
	'''

	availReg = ['SVR', 'BAG', 'Ridge', 'LS', 'LS+', 'RF', 'ADA', 'GBR']
	# Linear SVR, BaggingRegressor, Ridge, Lasso Regressor,
	# Lasso (With positive constraint), RandomForestRegressor,
	# AdaBoost Regressor, GradientBoostingRegressor

	savePath = 'tmpData/ensemble/audioFetAMultiStage/' + fetChoice + '/'

	if not os.path.exists(savePath):
		os.makedirs(savePath)

	bagAppend = 'B_' + str(bagVal)
	# Refers to the bagging value used
	dataAppend = 'D_'
	# Refers to the data index
	bagIDAppend = 'ID_'
	# Refers to the bagID
	mergeAppend = 'M_'
	# Refers to the merger used
	segmentAppend = 'S_'
	# Maybe create a function for this

	histListStageA = []
	histListStageB = []

	for j in range(len(fetList)):

		for c in range(len(segmentList)):
			fileName = savePath + fetList[j] + '_' + str(segmentList[c]) + '.p'

			print 'Reading data for:', fetList[j], segmentList[c]

			if (not os.path.isfile(fileName)):
				trainSet, X_train, Y_train = getMultiStageSet(vidNames, trueVal, fetChoice = fetChoice, clusterSize = segmentList[c])
				print '\nReading training data complete'
				testSet, X_test, Y_test = getMultiStageSet(vidNamesTest, trueVal, fetChoice = fetChoice, clusterSize = segmentList[c])
				print '\nReading testing data complete'
				dataList = [[trainSet, X_train, Y_train], [testSet, X_test, Y_test]]
				pickle.dump(dataList, open(fileName, 'wb'))
				print 'Data Dumping complete!'

			else:
				dataList = (pickle.load(open(fileName, 'rb')))
				trainSet, X_train, Y_train = dataList[0][0], dataList[0][1], dataList[0][2]
				testSet, X_test, Y_test = dataList[1][0], dataList[1][1], dataList[1][2]
				print 'Reading Done!'
			
			for d in range(len(mergeList)):

				regB = None

				if (mergeList[d] == 'LS'):
					regB = linear_model.Lasso(alpha = 1)
				elif (mergeList[d] == 'LS+'):
					regB = linear_model.Lasso(alpha = 1, positive = True)
				elif (mergeList[d] == 'Ridge'):
					regB = linear_model.Ridge(alpha = 10)
				elif (mergeList[d] == 'LinearSVR'):
					regB = LinearSVR(C = 30)
				elif (mergeList[d] == 'Poly2SVR'):
					regB = SVR(C = 30, kernel = 'poly', degree = 2)
				elif (mergeList[d] == 'Poly1SVR'):
					regB = SVR(C = 30, kernel = 'poly', degree = 1)
				elif (mergeList[d] == 'BAG_DecTree'):
					regB = BaggingRegressor(DecisionTreeRegressor(), n_estimators = 50, n_jobs = 4)
				elif (mergeList[d] == 'BAG_SVR'):
					regB = BaggingRegressor(LinearSVR(C = 30), n_estimators = 50, n_jobs = 4)
				else:
					print mergeList[d], 'not implemented yet!'
					continue

				for i in range(len(regList)):

					if (regList[i] not in availReg):
						continue

					if (regList[i] == 'SVR'):
						modelName = 'LinearSVR_'
						paramAppend = 'C_'
						# Gives the parameters for the regressor

						paramList = getParamValList(1e-3, 10, (numLimit/5), numLimit)

						for param in paramList:
							
							for k in range(numPerParam):

								clf = MultiStageRegressor()
								regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, fetList[j], \
													  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)
								regName = regName + '_' + segmentAppend + str(segmentList[c]) + '_' + mergeAppend + mergeList[d]
								clf.setName(regName)
								print regName

								regA = LinearSVR(C = param)
								clf.trainStageA(regA, numTargets, X_tr, Y_tr)
								clf.getStageAAcc(numTargets, X_test, Y_test)
								histListStageA.append([regName, clf.accA])

								if (accThreshold - clf.accA > (accThreshold/100)):
									break
								elif (clf.accA < accThreshold):
									continue

								clf.trainStageB(regB, numTargets, trainSet, Y_train)
								clf.getStageBAcc(numTargets, testSet, Y_test)
								histListStageB.append([regName, clf.accB])

								if (clf.accB > accThreshold):
									pickle.dump(clf, open(savePath + regName + '.p', 'wb'))

					elif (regList[i] == 'BAG'):
						modelName = 'BAG_DecTree_'
						paramAppend = 'N_'
						# Gives the number of estimators

						paramList = getParamValList(10, 10, 4, 4)

						for param in paramList:

							param = int(param)
							for k in range(numPerParam):

								clf = MultiStageRegressor()
								regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, fetList[j], \
													  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)
								regName = regName + '_' + segmentAppend + str(segmentList[c]) + '_' + mergeAppend + mergeList[d]
								print regName

								regA = BaggingRegressor(DecisionTreeRegressor(), n_estimators = param, n_jobs = 5)
								clf.trainStageA(regA, numTargets, X_tr, Y_tr)
								clf.getStageAAcc(numTargets, X_test, Y_test)
								histListStageA.append([regName, clf.accA])

								if (accThreshold - clf.accA > (accThreshold/100)):
									break
								elif (clf.accA < accThreshold):
									continue

								clf.trainStageB(regB, numTargets, trainSet, Y_train)
								clf.getStageBAcc(numTargets, testSet, Y_test)
								histListStageB.append([regName, clf.accB])

								if (clf.accB > accThreshold):
									pickle.dump(clf, open(savePath + regName + '.p', 'wb'))

					elif (regList[i] == 'Ridge'):
						modelName = 'Ridge_'
						paramAppend = 'Alpha_'
						# Gives the parameters for the regressor

						paramList = getParamValList(1e-3, 10, (numLimit/10), numLimit)

						for param in paramList:

							for k in range(numPerParam):

								clf = MultiStageRegressor()
								regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, fetList[j], \
													  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)
								regName = regName + '_' + segmentAppend + str(segmentList[c]) + '_' + mergeAppend + mergeList[d]
								print regName

								regA = linear_model.Ridge(alpha = param)
								clf.trainStageA(regA, numTargets, X_tr, Y_tr)
								clf.getStageAAcc(numTargets, X_test, Y_test)
								histListStageA.append([regName, clf.accA])

								if (accThreshold - clf.accA > (accThreshold/100)):
									break
								elif (clf.accA < accThreshold):
									continue

								clf.trainStageB(regB, numTargets, trainSet, Y_train)
								clf.getStageBAcc(numTargets, testSet, Y_test)
								histListStageB.append([regName, clf.accB])

								if (clf.accB > accThreshold):
									pickle.dump(clf, open(savePath + regName + '.p', 'wb'))

					elif (regList[i] == 'LS'):
						modelName = 'LS_'
						paramAppend = 'Alpha_'
						# Gives the parameters for the regressor

						paramList = getParamValList(1e-3, 10, (numLimit/10), numLimit)

						for param in paramList:

							for k in range(numPerParam):

								clf = MultiStageRegressor()
								regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, fetList[j], \
													  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)
								regName = regName + '_' + segmentAppend + str(segmentList[c]) + '_' + mergeAppend + mergeList[d]
								print regName

								regA = linear_model.Lasso(alpha = param)
								clf.trainStageA(regA, numTargets, X_tr, Y_tr)
								clf.getStageAAcc(numTargets, X_test, Y_test)
								histListStageA.append([regName, clf.accA])

								if (accThreshold - clf.accA > (accThreshold/100)):
									break
								elif (clf.accA < accThreshold):
									continue

								clf.trainStageB(regB, numTargets, trainSet, Y_train)
								clf.getStageBAcc(numTargets, testSet, Y_test)
								histListStageB.append([regName, clf.accB])

								if (clf.accB > accThreshold):
									pickle.dump(clf, open(savePath + regName + '.p', 'wb'))

					elif (regList[i] == 'LS+'):
						modelName = 'Lasso+_'
						paramAppend = 'Alpha_'
						# Gives the parameters for the regressor

						paramList = getParamValList(1e-3, 10, (numLimit/10), numLimit)

						for param in paramList:

							for k in range(numPerParam):

								clf = MultiStageRegressor()
								regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, fetList[j], \
													  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)
								regName = regName + '_' + segmentAppend + str(segmentList[c]) + '_' + mergeAppend + mergeList[d]
								clf.setName(regName)
								print regName

								regA = linear_model.Lasso(alpha = param, positive = True)
								clf.trainStageA(regA, numTargets, X_tr, Y_tr)
								clf.getStageAAcc(numTargets, X_test, Y_test)
								histListStageA.append([regName, clf.accA])

								if (accThreshold - clf.accA > (accThreshold/100)):
									break
								elif (clf.accA < accThreshold):
									continue

								clf.trainStageB(regB, numTargets, trainSet, Y_train)
								clf.getStageBAcc(numTargets, testSet, Y_test)
								histListStageB.append([regName, clf.accB])

								if (clf.accB > accThreshold):
									pickle.dump(clf, open(savePath + regName + '.p', 'wb'))

					elif (regList[i] == 'RF'):
						modelName = 'RF_'
						paramAppend = 'N_'
						# Gives the parameters for the regressor

						paramList = getParamValList(10, 10, 4, 4)

						for param in paramList:

							param = int(param)
							for k in range(numPerParam):

								clf = MultiStageRegressor()
								regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, fetList[j], \
													  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)
								regName = regName + '_' + segmentAppend + str(segmentList[c]) + '_' + mergeAppend + mergeList[d]
								clf.setName(regName)
								print regName

								regA = RandomForestRegressor(n_estimators = param, n_jobs = 5)
								clf.trainStageA(regA, numTargets, X_tr, Y_tr)
								clf.getStageAAcc(numTargets, X_test, Y_test)
								histListStageA.append([regName, clf.accA])

								if (accThreshold - clf.accA > (accThreshold/100)):
									break
								elif (clf.accA < accThreshold):
									continue

								clf.trainStageB(regB, numTargets, trainSet, Y_train)
								clf.getStageBAcc(numTargets, testSet, Y_test)
								histListStageB.append([regName, clf.accB])

								if (clf.accB > accThreshold):
									pickle.dump(clf, open(savePath + regName + '.p', 'wb'))

					elif (regList[i] == 'ADA'):
						modelName = 'ADA_DecTree_'
						paramAppend = 'N_'
						# Gives the parameters for the regressor

						paramList = getParamValList(10, 10, 4, 4)

						for param in paramList:

							param = int(param)
							for k in range(numPerParam):

								clf = MultiStageRegressor()
								regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, fetList[j], \
													  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)
								regName = regName + '_' + segmentAppend + str(segmentList[c]) + '_' + mergeAppend + mergeList[d]
								clf.setName(regName)
								print regName

								regA = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators = param)
								clf.trainStageA(regA, numTargets, X_tr, Y_tr)
								clf.getStageAAcc(numTargets, X_test, Y_test)
								histListStageA.append([regName, clf.accA])

								if (accThreshold - clf.accA > (accThreshold/100)):
									break
								elif (clf.accA < accThreshold):
									continue

								clf.trainStageB(regB, numTargets, trainSet, Y_train)
								clf.getStageBAcc(numTargets, testSet, Y_test)
								histListStageB.append([regName, clf.accB])

								if (clf.accB > accThreshold):
									pickle.dump(clf, open(savePath + regName + '.p', 'wb'))

					elif (regList[i] == 'GBR'):
						modelName = 'GBR_'
						paramAppend = 'N_'
						# Gives the parameters for the regressor

						paramList = getParamValList(10, 10, 4, 4)

						for param in paramList:

							param = int(param)
							for k in range(numPerParam):

								clf = MultiStageRegressor()
								regName, X_tr, Y_tr = getBaggedSet(modelName, bagAppend, dataAppend, fetList[j], \
													  paramAppend, param, bagIDAppend, k, X_train, Y_train, bagVal)
								regName = regName + '_' + segmentAppend + str(segmentList[c]) + '_' + mergeAppend + mergeList[d]
								clf.setName(regName)
								print regName

								regA = GradientBoostingRegressor(n_estimators = param, loss='lad')
								clf.trainStageA(regA, numTargets, X_tr, Y_tr)
								clf.getStageAAcc(numTargets, X_test, Y_test)
								histListStageA.append([regName, clf.accA])

								if (accThreshold - clf.accA > (accThreshold/100)):
									break
								elif (clf.accA < accThreshold):
									continue

								clf.trainStageB(regB, numTargets, trainSet, Y_train)
								clf.getStageBAcc(numTargets, testSet, Y_test)
								histListStageB.append([regName, clf.accB])

								if (clf.accB > accThreshold):
									pickle.dump(clf, open(savePath + regName + '.p', 'wb'))

					else:
						print regList[i], "not implemented yet!"

					print i, j
					print (histListStageA)
					print (histListStageB)

	print 'The history of training is as follows,'

	for x in histListStageA:
		print x[0], x[1]

	print '\n\n'

	for x in histListStageB:
		print x[0], x[1]

	return histListStageA, histListStageB


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

	actionChoice = 2

	if (actionChoice == 0):

		# regList = ['SVR', 'BAG', 'Ridge', 'LS', 'RF', 'ADA', 'GBR']
		regList = ['ADA', 'GBR']

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

	elif (actionChoice == 2):

		regList = ['SVR', 'BAG', 'Ridge', 'LS', 'RF', 'ADA', 'GBR']
		# mergeList = ['LS', 'LS+', 'Ridge', 'LinearSVR', 'BAG_SVR', 'Poly2SVR', 'Poly1SVR']
		mergeList = ['LS+', 'Ridge','BAG_SVR', 'Poly2SVR']
		fetList = ['AudioA_avg', 'AudioA_minmax']
		segmentList = [1, 3, 5, 7, 9, 11]
		fetChoice = 'AudioA_avg_minmax'

		createAudioMultiStageRegressorArmy(5, regList, vidNames, vidNamesTest, mergeList, fetList, segmentList, fetChoice, \
									   numLimit = 5, numPerParam = 2, accThreshold = 0.885, bagVal = 0.65)