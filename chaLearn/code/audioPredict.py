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
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.neural_network import MLPRegressor

np.set_printoptions(precision=2)

videoPath = '../training/download_train-val/trainFiles/'
vidNames = os.listdir(videoPath)
vidNames = [x for x in vidNames if x.endswith(".mp4")]

fileName = '../training/training_gt.csv'
trueVal = getTruthVal(fileName)

for i in xrange(len(vidNames)):
	vidNames[i] = vidNames[i].strip('.mp4')

vidNames = vidNames
splitVal = 0.9
vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
vidNames = vidNames[:int(splitVal*len(vidNames))]

fetChoice = 'AudioA_avg_cluster_4'
# X_train, Y_train = readData(vidNames, trueVal, feature = fetChoice, printFlag = True)
# X_test, Y_test = readData(vidNamesTest, trueVal, feature = fetChoice, printFlag = True)

# pickle.dump([X_train, Y_train, X_test, Y_test], open('tmpData/audioFetA/audioFet_' + fetChoice + '.p', 'wb'))
X_train, Y_train, X_test, Y_test = (pickle.load(open('tmpData/audioFetA/audioFet_' + fetChoice + '.p', 'rb')))

# X_train = X_train[:30000]
# Y_train = Y_train[:30000]
# X_test = X_test[:1000]
# Y_test = Y_test[:1000]

print len(X_train)

print X_train.shape, Y_train.shape
print X_test.shape, Y_test.shape
# p = pickle.load(open('tmpData/predictions/predA.p', 'rb'))
# generatePredFile(p)
# raw_input('FINISHED')

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_test = scaler.transform(X_test)
# X_test = np.nan_to_num(X_test)

Y_pred = np.zeros((X_test.shape[0], 5))

clfList = []

# modelChoice = 'NN'	# Ever so slightly better than constant
modelChoice = 'SVR'	# Gives out of bounds results
# modelChoice = 'RF'
# modelChoice = 'LGR'	# very poor results
# modelChoice = 'MISC'
# modelChoice = 'GBR'
# modelChoice = 'BAG'
# modelChoice = 'SKNN'

modelName, model_file_name = '', ''


print fetChoice, modelChoice

if (modelChoice == 'SVR'):
	modelName = 'audioFetA_SVR'
	model_file_name = 'tmpData/models/audioFetA_SVR'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1000.0, kernel = 'rbf'))
		clfList.append(SVR(C = 1.0, kernel = 'poly', degree = 2))
		# clfList.append(LinearSVR(C = 1000.0))
		# Parameter study for C
		clfList[i].fit(X_train, Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)

		Y_pred[Y_pred < 0] = 0
		Y_pred[Y_pred > 1] = 1

		print 'Predictions'
		print np.max(Y_pred[:,i])
		print np.min(Y_pred[:,i])
		print np.mean(Y_pred[:,i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'LGR'):
	modelName = 'audioFetA_LGR'
	model_file_name = 'tmpData/models/audioFetA_LGR'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(LinearSVR(C = 10.0))
		clfList.append(linear_model.Ridge(alpha = 0.1))
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

		model.fit(Y_tmp, Y_train[:,i], nb_epoch=15, batch_size = 128)
	
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

		print clfList[i].coef_
		print np.max(clfList[i].coef_)
		print np.min(clfList[i].coef_)
		print np.mean(clfList[i].coef_)

elif (modelChoice == 'ADA'):
	modelName = 'audioFetA_ADA'
	model_file_name = 'tmpData/models/audioFetA_ADA'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(AdaBoostRegressor(DecisionTreeRegressor(), n_estimators = 100))
		clfList[i].fit(X_train[i], Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test[i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'BAG'):
	modelName = 'audioFetA_BAG'
	model_file_name = 'tmpData/models/audioFetA_BAG_n50'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(BaggingRegressor(DecisionTreeRegressor(), n_estimators = 50, n_jobs = 4))
		# clfList.append(BaggingRegressor(linear_model.Ridge(alpha = 5000), n_estimators = 100, n_jobs = 2))
																	# No increase in performance from ridge (SInce it's not an unstable classifier)
		# clfList.append(linear_model.SGDRegressor())
		clfList[i].fit(X_train, Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)
		print np.max(Y_pred[:,i])
		print np.min(Y_pred[:,i])
		print np.mean(Y_pred[:,i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'GBR'):
	modelName = 'audioFetA_GBR'
	model_file_name = 'tmpData/models/audioFetA_GBR_n100'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0, loss='lad'))

		# clfList.append(linear_model.SGDRegressor())
		clfList[i].fit(X_train, Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)
		print np.max(Y_pred[:,i])
		print np.min(Y_pred[:,i])
		print np.mean(Y_pred[:,i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

elif (modelChoice == 'MISC'):
	modelName = 'audioFetA_MISC'
	model_file_name = 'tmpData/models/audioFetA_MISC'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(linear_model.Ridge(alpha = 5000))
		# clfList.append(linear_model.Lasso(alpha = 1e-4))
		# clfList.append(linear_model.SGDRegressor())
		clfList[i].fit(X_train, Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)

		print 'Predictions'
		print np.max(Y_pred[:,i])
		print np.min(Y_pred[:,i])
		print np.mean(Y_pred[:,i])
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

		print 'Coefficents'
		print clfList[i].coef_
		print np.max(clfList[i].coef_)
		print np.min(clfList[i].coef_)
		print np.mean(clfList[i].coef_)

elif (modelChoice == 'RF'):
	modelName = 'audioFetA_RF'
	model_file_name = 'tmpData/models/audioFetA_RF'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		clfList.append(RandomForestRegressor(n_estimators=511, n_jobs = 4))
		clfList[i].fit(X_train, Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])

		clfList[i].fit(X_train, Y_train[:,i])
		Y_tmp = clfList[i].predict(X_train)

		from keras.models import Sequential
		from keras.layers import Dense, Activation
		from keras.optimizers import SGD

		model = Sequential()

		model.add(Dense(1, input_dim=1, init='uniform'))
		model.add(Activation('sigmoid'))
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mean_absolute_error', optimizer=sgd)

		print 'NN model compiled. Training now...'

		model.fit(Y_tmp, Y_train[:,i], nb_epoch=30, batch_size = 32)
	
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

elif (modelChoice == 'NN'):

	modelName = 'audioFetA_NN'
	model_file_name = 'tmpData/models/audioFetA_NN'
		
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation
	from keras.optimizers import SGD

	model = Sequential()

	model.add(Dense(32, input_dim=1582, init='uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(16, input_dim=1582, init='uniform'))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.25))
	model.add(Dense(5, init='uniform'))
	model.add(Activation('sigmoid'))

	# Tested a few different variants, this setup works the best (Although not great)
	model.compile(loss='mean_absolute_error', optimizer='rmsprop')

	print 'NN model compiled. Training now...'

	model.fit(X_train, Y_train, nb_epoch=15, batch_size = 128, validation_data=(X_test, Y_test))
	score = model.evaluate(X_test, Y_test, batch_size = 32)

	Y_pred = model.predict(X_test)

elif (modelChoice == 'SKNN'):
	modelName = 'audioFetA_SKNN'
	model_file_name = 'tmpData/models/audioFetA_SKNN'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'rbf'))
		# clfList.append(linear_model.Ridge(alpha = 1))
		clfList.append(MLPRegressor(hidden_layer_sizes = (32,)))
		# clfList.append(linear_model.SGDRegressor())
		clfList[i].fit(X_train, Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)
		print np.corrcoef(Y_pred[:,i], Y_test[:,i])
		print clfList[i].coefs_
		print np.max(clfList[i].coefs_)
		print np.min(clfList[i].coefs_)
		print np.mean(clfList[i].coefs_)

# Scale transform data, save the scaler for later use
# Save the trained models and predictions
pickle.dump(clfList, open(model_file_name + fetChoice + '.p', 'wb'))
# pickle.dump(Y_pred, open(model_file_name, 'rb'))

print Y_test

print Y_pred
print Y_pred.max(0)
print Y_pred.min(0)
print np.mean(Y_pred, axis=0)

evaluateTraits(Y_pred, Y_test)