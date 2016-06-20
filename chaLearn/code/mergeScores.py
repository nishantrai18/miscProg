import cPickle as pickle
import numpy as np
from readVideo import *
import random
import sklearn
from sklearn import linear_model
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

def getSortedFeatures(fetList, size = 15):
	if (len(fetList) == 0):
		avg = 0.5
		fetList = np.append(fetList, ([avg]*(size-len(fetList))))
	elif (len(fetList) < size):
		avg = np.mean(fetList)
		fetList = np.append(fetList, ([avg]*(size-len(fetList))))
	elif (len(fetList) > size):
		for i in range(len(fetList)-size):
			fetList[-2] = (fetList[-2]+fetList[-1])/2.0
			fetList = np.delete(fetList, -1)
			newVal = fetList[-1]
			fetList = np.delete(fetList, -1)
			fetList = np.insert(fetList, 0, newVal)
	sortFet = np.sort(fetList)
	return sortFet

def weightedModelFit(X, Y):
	weights = np.zeros((X.shape[1]))
	for i in xrange(X.shape[0]):
		weights = weights + np.abs(X[i] - Y[i])
	weights = 1.0/weights
	# weights = np.power(weights, 0.5)
	weights = weights/np.sum(weights)
	return weights

def weightedModelPredict(X, weights):
	pred = np.zeros(X.shape[0])
	for i in xrange(X.shape[0]):
		pred[i] = np.dot(weights, X[i])
	return pred

videoPath = '../training/download_train-val/trainFiles/'
vidNames = os.listdir(videoPath)
vidNames = [x for x in vidNames if x.endswith(".mp4")]

fileName = '../training/training_gt.csv'
trueVal = getTruthVal(fileName)

for i in xrange(len(vidNames)):
	vidNames[i] = vidNames[i].strip('.mp4')

choice = 'C'
splitVal = 0.9
vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
vidNames = vidNames[:int(splitVal*len(vidNames))]

# trainData = pickle.load(open('tmpData/predictions/predListvisualFetA_BasicConv_Augmented_32_64_256_train.p', 'rb'))
trainData = pickle.load(open('tmpData/predictions/predListvisualFetC_Conv_Augmented_32_64_256_train.p', 'rb'))
X_train, Y_train = [], []

for i in range(5):
	X_train.append([])
	for k in trainData.keys():
		if (len(trainData[k]) == 0):
			continue
		X_train[i].append(getSortedFeatures(trainData[k][:,i]))
		if (i == 0):
			# Do this only once
			Y_train.append(trueVal[k])
	X_train[i] = np.array(X_train[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# testData = pickle.load(open('tmpData/predictions/predListvisualFetA_BasicConv_Augmented_32_64_256_test.p', 'rb'))
testData = pickle.load(open('tmpData/predictions/predListvisualFetC_Conv_Augmented_32_64_256_test.p', 'rb'))
X_test, Y_test = [], []

for i in range(5):
	X_test.append([])
	for k in testData.keys():
		if (len(testData[k]) == 0):
			continue
		X_test[i].append(getSortedFeatures(testData[k][:,i]))
		if (i == 0):
			# Do this only once
			Y_test.append(trueVal[k])
	X_test[i] = np.array(X_test[i])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

print X_train.shape, Y_train.shape
print X_test.shape, Y_test.shape

Y_pred = np.zeros((X_test.shape[1], 5))

clfList = []

# modelChoice = 'NN'		# Poor performance
# modelChoice = 'SVR'		# Comparable results to Lasso
modelChoice = 'LS'			# Best performer
# modelChoice = 'RF'
# modelChoice = 'ADA'
# modelChoice = 'WGT'		# Performs slightly worse than simple average

modelName, model_file_name = '', ''

if (modelChoice == 'LS'):
	modelName = 'mergeScore_Fet' + choice + '_LS'
	model_file_name = 'tmpData/models/mergeScore_Fet' + choice + '_LS'
	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(linear_model.Lasso(alpha = 2e-4, positive = True, max_iter = 5000))
		# Parameter study for C
		clfList[i].fit(X_train[i], Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		# print clfList[i].predict(X_test[i])[:100]
		print clfList[i].coef_
		Y_pred[:,i] = clfList[i].predict(X_test[i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'SVR'):
	modelName = 'mergeScore_Fet' + choice + '_SVR'
	model_file_name = 'tmpData/models/mergeScore_Fet' + choice + '_SVR'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(LinearSVR(C = 0.01))
		# Parameter study for C
		clfList[i].fit(X_train[i], Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test[i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'RF'):
	modelName = 'mergeScore_Fet' + choice + '_RF'
	model_file_name = 'tmpData/models/mergeScore_Fet' + choice + '_RF'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(RandomForestRegressor(random_state=0, n_estimators=11))
		clfList[i].fit(X_train[i], Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test[i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'ADA'):
	modelName = 'mergeScore_Fet' + choice + '_ADA'
	model_file_name = 'tmpData/models/mergeScore_Fet' + choice + '_ADA'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=50))
		clfList[i].fit(X_train[i], Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test[i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'WGT'):
	modelName = 'mergeScore_Fet' + choice + '_WGT'
	model_file_name = 'tmpData/models/mergeScore_Fet' + choice + '_WGT'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(weightedModelFit(X_train[i], Y_train[:,i]))
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = weightedModelPredict(X_test[i], clfList[i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'NN'):
	modelName = 'mergeScore_Fet' + choice + '_NN'
	model_file_name = 'tmpData/models/mergeScore_Fet' + choice + '_NN'
		
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation
	from keras.optimizers import SGD

	for i in range(5):
		print 'Currently training the', i, 'th NN'
	
		model = Sequential()

		model.add(Dense(5, input_dim=15, init='uniform'))
		model.add(Activation('relu'))
		model.add(Dense(1, init='uniform'))
		model.add(Activation('sigmoid'))

		# Tested a few different variants, this setup works the best (Although not great)
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mean_absolute_error', optimizer=sgd)

		print 'NN model compiled. Training now...'

		model.fit(X_train[i], Y_train[:,i], nb_epoch=10, batch_size = 32, validation_data=(X_test[i], Y_test[:,i]))
		score = model.evaluate(X_test[i], Y_test[:,i], batch_size = 32)

		Y_pred[:,i] = model.predict(X_test[i])[:1]
		clfList.append(model)

# Scale transform data, save the scaler for later use
# Save the trained models and predictions
pickle.dump(clfList, open(model_file_name + '.p', 'wb'))
# pickle.dump(Y_pred, open(model_file_name, 'wb'))

# print Y_test
# print Y_pred
# print Y_pred.max(0)
# print Y_pred.min(0)
print np.mean(Y_pred, axis=0)

evaluateTraits(Y_pred, Y_test)