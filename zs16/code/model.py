from sklearn.linear_model import Ridge
from reader import *
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support

import sklearn
from sklearn.svm import SVR, LinearSVR, LinearSVC, SVC
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier

import random

def evaluatePred(Y_pred, Y_true):
	mmse = np.sum((Y_pred - Y_true)**2)/((Y_pred.shape[0]) * (1.0))
	bmse = Y_true.var()
	print mmse, bmse
	print ((bmse - mmse)/bmse)

def procedureA():
	# Trains and generates a prediction file
	# Uses hard heuristic for buy_or_not

	popFlag = True
	X, Y = getDataXY(currYearFlag = False, popFlag = popFlag)
	X, Y = shuffle(X, Y, random_state = 0)

	if popFlag:
		encoder = oneHot(X[:, 2:])
		Xt = encoder.transform(X[:, 2:])
		Xt = np.hstack((X[:,:2].reshape(-1, 1), Xt))
	else:
		encoder = oneHot(X)
		Xt = encoder.transform(X)

	buySet = set()
	for i in range(X.shape[0]):
		tmpTup = (X[i][0], X[i][2])
		buySet.add(tmpTup)
	# Y_buy = [1] * Xt.shape[0]

	min_max_scaler = preprocessing.MinMaxScaler()
	# Xt = min_max_scaler.fit_transform(Xt)

	split = 0.9
	X_train, X_test = Xt[:(int(Xt.shape[0]*split)),:], Xt[int(Xt.shape[0]*split):, :]
	Y_train, Y_test = Y[:(int(Y.shape[0]*split)),:], Y[int(Y.shape[0]*split):, :]
	Y_train = Y_train.ravel()
	Y_test = Y_test.ravel()

	print X_train.shape
	print X_test.shape

	# clf = Ridge(alpha = 100)
	# clf = SVR(C = 10.0, kernel = 'poly', degree = 2)
	# clf = LinearSVR(C = 1.0)
	clf = BaggingRegressor(DecisionTreeRegressor(), n_estimators = 125, n_jobs = 4)
	# clf = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators = 100)
	# clf = DecisionTreeRegressor()
	# clf = RandomForestRegressor(random_state = 0, n_estimators = 200, n_jobs = 4)
	clf.fit(X_train, Y_train.ravel())

	Y_pred = clf.predict(X_test)
	evaluatePred(Y_pred, Y_test)

	return clf, encoder, min_max_scaler
	# generatePredFile(buySet, clf, encoder)

def getNegativeSamples(sizeList, popList, buySet, encoder):
	# Get negative samples for training
	# HEURISTIC: Only penalize those which are surely going to be erroneous
	# Choose only 10% of the total possible samples

	negX = []
	negY = []

	for i in range(1, sizeList[0]):
		for j in range(1, sizeList[1]):
			for k in range(1, sizeList[2]):
				tmpVal = (i, k)
				if (tmpVal not in buySet):
					if (random.randint(0, 2) <= 1):
						fetList = [i, -1, j, k]
						tmpKey = (fetList[0], fetList[2])
						if (tmpKey in popList):
							fetList[0] = popList[tmpKey]
							fetList[1] = 0
						else:
							fetList[0] = 0
							fetList[1] = 1
						fetList = np.array(fetList)
						negX.append(fetList)

	negX = np.array(negX)
	negY = [0] * negX.shape[0]
	negY = np.array(negY)
	negY = negY.reshape(-1, 1)

	print negX.shape
	print negY.shape

	return negX, negY

def procedureB(paramC = 1.0):
	# Predicting buy_or_not in a better way, logistic regression with weights on samples
	
	popFlag = True # DON'T MODIFY, LOTS OF SAMPLES

	X, Y = getDataXY(currYearFlag = True, popFlag = popFlag)

	Y = [1] * X.shape[0]
	Y = np.array(Y)
	Y = Y.reshape(-1, 1)

	if popFlag:
		encoder = oneHot(X[:, 2:])
		Xt = encoder.transform(X[:, 2:])
		Xt = np.hstack((X[:,:2].reshape(-1, 1), Xt))
	else:
		encoder = oneHot(X)
		Xt = encoder.transform(X)

	# Golden features
	# Xt = getGoldenX(Xt, 2, 2 + encoder.n_values_[0], 2 + encoder.n_values_[0], 2 + encoder.n_values_[0] + encoder.n_values_[1])

	tX, tY = getDataXY(currYearFlag = True, popFlag = False)
	# To get the index of the hospitals
	buySet = set()
	for i in range(tX.shape[0]):
		tmpTup = (tX[i][0], tX[i][2])
		buySet.add(tmpTup)

	maxList = np.max(tX, axis = 0)

	popList = getHospitalProfile()
	negX, negY = getNegativeSamples(maxList, popList, buySet, encoder)

	if popFlag:
		negXt = encoder.transform(negX[:, 2:])
		negXt = np.hstack((negX[:,:2].reshape(-1, 1), negXt))
	else:
		negXt = encoder.transform(negX)

	Xt = np.vstack((Xt, negXt))

	print Y.shape
	print negY.shape

	Y = np.vstack((Y, negY))
	Xt, Y = shuffle(Xt, Y, random_state = 0)

	print Xt.shape
	print Y.shape

	min_max_scaler = preprocessing.MinMaxScaler()
	# Xt = min_max_scaler.fit_transform(Xt)

	split = 0.9
	X_train, X_test = Xt[:(int(Xt.shape[0]*split)),:], Xt[int(Xt.shape[0]*split):, :]
	Y_train, Y_test = Y[:(int(Y.shape[0]*split)),:], Y[int(Y.shape[0]*split):, :]
	Y_train = Y_train.ravel()
	Y_test = Y_test.ravel()

	print X_train.shape
	print X_test.shape

	# clf = LogisticRegression(C = paramC, class_weight = {0:1, 1:10}, random_state = 0)
	# clf = RandomForestClassifier(class_weight = {0:1, 1:50}, random_state = 0, n_estimators = 125, n_jobs = 4)
	clf = RandomForestClassifier(class_weight = 'auto', random_state = 0, n_estimators = 125, n_jobs = 4)
	# clf = Ridge(alpha = 100)
	# clf = SVR(C = 10.0, kernel = 'poly', degree = 2)
	# clf = LinearSVC(C = 1.0, class_weight = {0:1, 1:7}, random_state = 0)
	# clf = BaggingClassifier(LogisticRegression(C = paramC, class_weight = {0:1, 1:10}, random_state = 0), n_estimators = 50, n_jobs = 4)
	# clf = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators = 100)
	# clf = DecisionTreeRegressor()
	# clf = RandomForestRegressor(random_state = 0, n_estimators = 200, n_jobs = 4)
	clf.fit(X_train, Y_train.ravel())

	Y_pred = clf.predict(X_test)

	# for t in zip(clf.predict_proba(X_test), Y_test):
	# 	print t

	print clf.score(X_test, Y_test)

	print precision_recall_fscore_support(Y_test, Y_pred, average = 'binary')

	# print clf.coef_

	evaluatePred(Y_pred, Y_test)

	# generatePredFile(buySet, clf, encoder)
	return clf, encoder, min_max_scaler

def procedureC():
	clfRevenue, encRev, scalerRev = procedureA()
	clfBuy, encBuy, scalerBuy = procedureB(paramC = 0.1)
	generatePredFileC(clfBuy, clfRevenue, encBuy, encRev, scalerBuy, scalerRev)


def visualize():
	# Get information about district, instrument, etc wise information

	district = {}

	X, Y = getDataXY()

	for i in range(X.shape[0]):
		if (X[i][3] not in district):
			district[X[i][3]] = []
		district[X[i][3]].append(Y[i][0])

	print len(district)

	for i in district.keys():
		print '\n\nVALUES : ', i, '\n\n'
		print district[i]

if __name__ == "__main__":

	random.seed(0)
	# For reproducibility

	procedureC()

	# for c in [0.1, 1, 10, 100, 1000, 10000]:
	# 	procedureB(c)
	# # visualize()
