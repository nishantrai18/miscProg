import cPickle as pickle
import numpy as np
from readVideo import *
import random
import sklearn
from sklearn.svm import SVR, LinearSVR
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

videoPath = '../training/download_train-val/trainFiles/'
vidNames = os.listdir(videoPath)
vidNames = [x for x in vidNames if x.endswith(".mp4")]

fileName = '../training/training_gt.csv'
trueVal = getTruthVal(fileName)

for i in xrange(len(vidNames)):
	vidNames[i] = vidNames[i].strip('.mp4')

splitVal = 0.9
vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
vidNames = vidNames[:int(splitVal*len(vidNames))]

X_train, Y_train = readData(vidNames, trueVal, feature = 'B')
# X_test = X_test.reshape(X_test.shape[0], 3, row, col)

X_test, Y_test = readData(vidNamesTest, trueVal, feature = 'B')
# X_test = X_test.reshape(X_test.shape[0], 3, row, col)

print X_train.shape, Y_train.shape
print X_test.shape, Y_test.shape
# p = pickle.load(open('tmpData/predictions/predA.p', 'rb'))
# generatePredFile(p)
# raw_input('FINISHED')
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_test = scaler.transform(X_test)
Y_pred = np.zeros((X_test.shape[0], 5))

clfList = []

# modelChoice = 'NN'	# Ever so slightly better than constant
# modelChoice = 'SVR'	# Gives out of bounds results
# modelChoice = 'RF'
# modelChoice = 'LGR'	# very poor results
modelChoice = 'MISC'
# modelChoice = 'BAG'

if (modelChoice == 'SVR'):
	modelName = 'visualFetB_SVR'
	model_file_name = 'tmpData/models/visualFetB_SVR_C1.0'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(LinearSVR(C = 10000.0))
		# Parameter study for C
		clfList[i].fit(X_train, Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'LGR'):
	modelName = 'visualFetB_LGR'
	model_file_name = 'tmpData/models/visualFetB_LGR'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		clfList.append(SVR(C = 100.0, kernel = 'rbf'))
		# clfList.append(LinearSVR(C = 10000.0))
		# Parameter study for C
		clfList[i].fit(X_train, Y_train[:,i])
		Y_tmp = clfList[i].predict(X_train)

		from keras.models import Sequential
		from keras.layers import Dense, Activation
		from keras.optimizers import SGD

		model = Sequential()

		model.add(Dense(1, input_dim=1, init='uniform'))
		model.add(Activation('sigmoid'))
		sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mean_absolute_error', optimizer=sgd)

		print 'NN model compiled. Training now...'

		model.fit(Y_tmp, Y_train[:,i], nb_epoch=15, batch_size = 32)
	
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)
		print 'OLD'
		print Y_pred[:,i]
		tmp = model.predict(Y_pred[:,i].reshape(-1, 1))
		# print tmp.shape
		Y_pred[:,i] = model.predict(Y_pred[:,i].reshape(-1, 1))[:,0]

		print 'NEW'
		print Y_pred[:,i]
		print 'SUPPOSED to be'
		print Y_test[:,i]
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'MISC'):
	modelName = 'visualFetB_MISC'
	model_file_name = 'tmpData/models/visualFetB_MISC'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(linear_model.Ridge(alpha = 0.05))
		clfList[i].fit(X_train, Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'BAG'):
	modelName = 'visualFetB_BAG'
	model_file_name = 'tmpData/models/visualFetB_BAG'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(BaggingRegressor(DecisionTreeRegressor(), n_estimators = 40, n_jobs = 4))
		# clfList.append(BaggingRegressor(linear_model.Ridge(alpha = 5)))		
		# clfList.append(linear_model.SGDRegressor())
		clfList[i].fit(X_train, Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)
		print np.max(Y_pred[:,i])
		print np.min(Y_pred[:,i])
		print np.mean(Y_pred[:,i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'RF'):
	modelName = 'visualFetB_RF'
	model_file_name = 'tmpData/models/visualFetB_RF'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(RandomForestRegressor(random_state=0, n_estimators=11))
		clfList[i].fit(X_train, Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

		# clfList[i].fit(X_train, Y_train[:,i])
		# Y_tmp = clfList[i].predict(X_train)

		# from keras.models import Sequential
		# from keras.layers import Dense, Activation
		# from keras.optimizers import SGD

		# model = Sequential()

		# model.add(Dense(1, input_dim=1, init='uniform'))
		# model.add(Activation('sigmoid'))
		# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		# model.compile(loss='mean_absolute_error', optimizer=sgd)

		# print 'NN model compiled. Training now...'

		# model.fit(Y_tmp, Y_train[:,i], nb_epoch=15, batch_size = 32)
	
		# print 'Model Trained. Prediction in progress'
		# Y_pred[:,i] = clfList[i].predict(X_test)
		# print 'OLD'
		# print Y_pred[:,i]
		# tmp = model.predict(Y_pred[:,i].reshape(-1, 1))
		# # print tmp.shape
		# Y_pred[:,i] = model.predict(Y_pred[:,i].reshape(-1, 1))[:,0]

		# print 'NEW'
		# print Y_pred[:,i]
		# print 'SUPPOSED to be'
		# print Y_test[:,i]
		# print np.corrcoef(Y_pred[:,i], Y_test[:,i])


elif (modelChoice == 'NN'):

	modelName = 'visualFetB_NN'
	model_file_name = 'tmpData/models/visualFetB_SVR_C1.0'
		
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation
	from keras.optimizers import SGD

	model = Sequential()

	model.add(Dense(64, input_dim=136, init='uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(32, init='uniform'))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.25))
	model.add(Dense(5, init='uniform'))
	model.add(Activation('sigmoid'))

	# Tested a few different variants, this setup works the best (Although not great)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])

	print 'NN model compiled. Training now...'

	model.fit(X_train, Y_train, nb_epoch=15, batch_size = 32, validation_data=(X_test, Y_test))
	score = model.evaluate(X_test, Y_test, batch_size = 32)

	Y_pred = model.predict(X_test)

# Scale transform data, save the scaler for later use
# Save the trained models and predictions
pickle.dump(clfList, open(model_file_name + '.p', 'wb'))
# pickle.dump(Y_pred, open(model_file_name, 'rb'))

# pickle.dump(predDict, open(predFileName + '.p', 'wb'))

print Y_test

print Y_pred
print Y_pred.max(0)
print Y_pred.min(0)
print np.mean(Y_pred, axis=0)

evaluateTraits(Y_pred, Y_test)