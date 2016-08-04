import cPickle as pickle
import numpy as np
from readVideo import *
from evaluateModel import *
import random
import sklearn
from sklearn import linear_model
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

def weightedModelFit(X, Y):
	weights = np.zeros((X.shape[1]))
	for i in xrange(X.shape[0]):
		weights = weights + np.abs(X[i] - Y[i])
		# weights = weights + np.abs(1)
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

# choice = 'C'
# choice = 'A'
choice = 'AudioA_avg_cluster_4'
# choice = '_AudioA'
# choice = '' #For standard audio
splitVal = 0.9
origVidNames = []
vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
vidNames = vidNames[:int(splitVal*len(vidNames))]

# trainData = pickle.load(open('tmpData/predictions/predListvisualFetA_BasicConv_Augmented_32_64_256_train.p', 'rb'))
# trainData = pickle.load(open('tmpData/predictions/predListvisualFetC_Conv_Augmented_32_64_256_train' + str(splitVal) +'.p', 'rb'))
trainData = pickle.load(open('tmpData/predictions/predListaudioFetA_BAG_n50' + choice + '_train' + str(splitVal) +'.p', 'rb'))
# trainData = pickle.load(open('tmpData/predictions/predListvisualFetF_VGG_5_128_4096_avg_train' + str(splitVal) +'.p', 'rb'))
# trainData = pickle.load(open('tmpData/predictions/predListvisualFetC_Conv_48_96_256_train' + str(splitVal) +'.p', 'rb'))
X_train, Y_train = [], []

for i in range(5):
	X_train.append([])
	for k in trainData.keys():
		# print trainData[k]
		if (len(trainData[k]) == 0):
			continue
		X_train[i].append(getSortedFeatures(trainData[k][:,i]))
		# X_train[i].append(getCompleteSortedFeatures(trainData[k], sortFlag = True))
		if (i == 0):
			# Do this only once
			Y_train.append(trueVal[k])
			# origVidNames.append(k)

	X_train[i] = np.array(X_train[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# testData = pickle.load(open('tmpData/predictions/predListvisualFetA_BasicConv_Augmented_32_64_256_test.p', 'rb'))
# testData = pickle.load(open('tmpData/predictions/predListvisualFetC_Conv_Augmented_32_64_256_test' + str(splitVal) +'.p', 'rb'))
testData = pickle.load(open('tmpData/predictions/predListaudioFetA_BAG_n50' + choice + '_test' + str(splitVal) +'.p', 'rb'))
# testData = pickle.load(open('tmpData/predictions/predListvisualFetC_Conv_48_96_256_test' + str(splitVal) +'.p', 'rb'))
# testData = pickle.load(open('tmpData/predictions/predListvisualFetF_VGG_5_128_4096_avg_test' + str(splitVal) +'.p', 'rb'))
X_test, Y_test = [], []

for i in range(5):
	X_test.append([])
	for k in testData.keys():
		if (len(testData[k]) == 0):
			continue
		X_test[i].append(getSortedFeatures(testData[k][:,i]))
		# X_test[i].append(getCompleteSortedFeatures(testData[k], sortFlag = True))
		if (i == 0):
			# Do this only once
			Y_test.append(trueVal[k])
			origVidNames.append(k)
	X_test[i] = np.array(X_test[i])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

print X_train.shape, Y_train.shape
print X_test.shape, Y_test.shape

Y_pred = np.zeros((X_test.shape[1], 5))

clfList = []

# modelChoice = 'NN'		# Poor performance
# modelChoice = 'SVR'		# Comparable results to Lasso
modelChoice = 'LS'			# Best performer, Ridge also performs well
# modelChoice = 'RF'
# modelChoice = 'ADA'
# modelChoice = 'WGT'		# Performs slightly worse than simple average
# modelChoice = 'BAG'
# modelChoice = 'Ensemble_LS'
# modelChoice = 'Ensemble_compFet_LS'
# modelChoice = 'Ensemble_compFet_BAG'
# modelChoice = 'Ensemble_compFet_SVR'
# modelChoice = 'Ensemble_WGT'
# modelChoice = 'Ensemble_BAG'
# modelChoice = 'Ensemble_ADA'
# modelChoice = 'Ensemble_RF'

# RF, ADA, BAG, LS are good diverse results

# choice = 'AudioA'
# append = '_orig'
# append = '_48_96'
append = ''
modelName, model_file_name = '', ''
predFileName = ''

if (modelChoice == 'LS'):
	modelName = 'finaleMergeScore_Fet' + choice + append + '_LS'
	model_file_name = 'tmpData/models/finaleMergeScore_Fet' + choice + append + '_LS'
	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(linear_model.Ridge(alpha = 10))
		# clfList.append(linear_model.Lasso(alpha = 2e-3, max_iter = 5000))
		# Parameter study for C
		clfList[i].fit(X_train[i], Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		# print clfList[i].predict(X_test[i])[:100]
		print clfList[i].coef_
		Y_pred[:,i] = clfList[i].predict(X_test[i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'SVR'):
	modelName = 'finaleMergeScore_Fet' + choice + append + '_SVR'
	model_file_name = 'tmpData/models/finaleMergeScore_Fet' + choice + append + '_SVR'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		clfList.append(SVR(C = 10, kernel = 'poly', degree = 2, coef0 = 0.1))
		# clfList.append(LinearSVR(C = 0.01))
		# Parameter study for C
		clfList[i].fit(X_train[i], Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test[i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'RF'):
	modelName = 'mergeScore_Fet' + choice + append + '_RF'
	model_file_name = 'tmpData/models/mergeScore_Fet' + choice + append + '_RF'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(RandomForestRegressor(random_state=0, n_estimators=11))
		clfList[i].fit(X_train[i], Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test[i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'ADA'):
	modelName = 'finaleMergeScore_Fet' + choice + append + '_ADA'
	model_file_name = 'tmpData/models/finaleMergeScore_Fet' + choice + append + '_ADA'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4), n_estimators = 100))
		clfList[i].fit(X_train[i], Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test[i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'BAG'):
	modelName = 'finaleMergeScore_Fet' + choice + append + '_BAG'
	model_file_name = 'tmpData/models/finaleMergeScore_Fet' + choice + append + '_BAG'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(BaggingRegressor(DecisionTreeRegressor(max_depth = 5), n_estimators = 100, n_jobs = 4))
		# clfList.append(BaggingRegressor(SVR(C = 10, kernel = 'rbf', degree = 2, coef0 = 1), n_estimators = 50, n_jobs = 4))
		# clfList.append(BaggingRegressor(linear_model.Ridge(alpha = 10), n_estimators = 50, n_jobs = 4))		
		# clfList.append(linear_model.SGDRegressor())
		clfList[i].fit(X_train[i], Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test[i])
		print np.max(Y_pred[:,i])
		print np.min(Y_pred[:,i])
		print np.mean(Y_pred[:,i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'WGT'):
	modelName = 'finaleMergeScore_Fet' + choice + append + '_WGT'
	model_file_name = 'tmpData/models/finaleMergeScore_Fet' + choice + append + '_WGT'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(weightedModelFit(X_train[i], Y_train[:,i]))
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = weightedModelPredict(X_test[i], clfList[i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'NN'):
	modelName = 'mergeScore_Fet' + choice + append + '_NN'
	model_file_name = 'tmpData/models/mergeScore_Fet' + choice + append + '_NN'
		
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation
	from keras.optimizers import SGD

	for i in range(5):
		print 'Currently training the', i, 'th NN'
	
		model = Sequential()

		model.add(Dense(5, input_dim=15, init='uniform'))
		model.add(Activation('tanh'))
		model.add(Dense(1, init='uniform'))
		model.add(Activation('sigmoid'))

		# Tested a few different variants, this setup works the best (Although not great)
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mean_absolute_error', optimizer=sgd)

		print 'NN model compiled. Training now...'

		model.fit(X_train[i], Y_train[:,i], nb_epoch=10, batch_size = 32, validation_data=(X_test[i], Y_test[:,i]))
		score = model.evaluate(X_test[i], Y_test[:,i], batch_size = 32)

		Y_pred[:,i] = model.predict(X_test[i])[:,0]
		clfList.append(model)

elif (modelChoice == 'Ensemble_WGT'):

	modelName = 'mergeScore_Fet' + choice + append + '_WGT'
	model_file_name = 'tmpData/models/mergeScore_Fet' + choice + append + '_WGT'
	predFileName = 'tmpData/predictions/mergeScore_Fet' + choice + append +'_WGT'
	
	clfList = pickle.load(open(model_file_name + '.p', 'rb'))
	X = X_test
	Y = np.zeros((X.shape[1], 5))

	for i in range(5):
		print 'Currently predicting using the', i, 'th regressor'
		Y[:,i] = weightedModelPredict(X[i], clfList[i])

	Y[Y < 0] = 0
	Y[Y > 1] = 1

	predDict = {}
	for i in range(len(origVidNames)):
		fileName = origVidNames[i]
		predDict[fileName] = Y[i]

elif ('Ensemble' in modelChoice):

	modelName = 'finaleMergeScore_Fet' + choice + append + modelChoice.strip('Ensemble')
	model_file_name = 'tmpData/models/finaleMergeScore_Fet' + choice +  append + modelChoice.strip('Ensemble')
	predFileName = 'tmpData/predictions/finaleMergePred_Fet' + choice + append + modelChoice.strip('Ensemble')
	
	print model_file_name

	clfList = pickle.load(open(model_file_name + '.p', 'rb'))

	# X = np.concatenate((X_train, X_test), axis = 1)
	X = X_test
	Y = np.zeros((X.shape[1], 5))

	print X.shape, X_train.shape, X_test.shape
	
	for i in range(5):
		print 'Currently predicting using the', i, 'th regressor'
		Y[:,i] = clfList[i].predict(X[i])

	Y[Y < 0] = 0
	Y[Y > 1] = 1

	predDict = {}
	for i in range(len(origVidNames)):
		fileName = origVidNames[i]
		predDict[fileName] = Y[i]

# Scale transform data, save the scaler for later use
# Save the trained models and predictions

if (predFileName != ''):
	pickle.dump(predDict, open(predFileName + '.p', 'wb'))

	print len(predDict)

	# print Y_test
	# print Y_pred
	# print Y_pred.max(0)
	# print Y_pred.min(0)
	print np.mean(Y, axis=0)
	# Y_true = np.concatenate((Y_train, Y_test), axis = 0)
	Y_true = Y_test
	# print Y_true
	# print Y
	print Y.max(0)
	print Y.min(0)
	evaluateTraits(Y, Y_true)

else:
	# print Y_test
	# print Y_pred
	print Y_pred.max(0)
	print Y_pred.min(0)
	print np.mean(Y_pred, axis=0)
	evaluateTraits(Y_pred, Y_test)
	pickle.dump(clfList, open(model_file_name + '.p', 'wb'))
