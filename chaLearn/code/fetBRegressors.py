import cPickle as pickle
import numpy as np
from readVideo import *
import random
import sklearn
from sklearn.svm import SVR, LinearSVR
from sklearn import preprocessing

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
scaler = preprocessing.StandardScaler().fit(X_train)
X_test = scaler.transform(X_test)
Y_pred = np.zeros((X_test.shape[0], 5))

clfList = []

modelChoice = 'NN'
# modelChoice = 'SVR'
# modelChoice = 'RF'

if (modelChoice == 'SVR'):
	modelName = 'visualFetB_SVR'
	model_file_name = 'tmpData/models/visualFetB_SVR_C1.0'

	for i in range(5):
		print 'Currently training the', i, 'th regressor'
		# clfList.append(SVR(C = 1.0, kernel = 'linear'))
		clfList.append(LinearSVR(C = 100.0))
		# Parameter study for C
		clfList[i].fit(X_train, Y_train[:,i])
		print 'Model Trained. Prediction in progress'
		Y_pred[:,i] = clfList[i].predict(X_test)

elif (modelChoice == 'NN'):

	modelName = 'visualFetB_NN'
	model_file_name = 'tmpData/models/visualFetB_SVR_C1.0'
		
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation
	from keras.optimizers import SGD

	model = Sequential()

	model.add(Dense(64, input_dim=136, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(5, init='uniform'))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])

	print 'NN model compiled. Training now...'

	model.fit(X_train, Y_train, nb_epoch=20, batch_size = 64)
	score = model.evaluate(X_test, Y_test, batch_size = 64)

	Y_pred = model.predict(X_test)

# Scale transform data, save the scaler for later use
# Save the trained models and predictions
# pickle.dump([clfList, scaler], open(model_file_name + '.p', 'wb'))
# pickle.dump(Y_pred, open(model_file_name, 'rb'))

print Y_test

print Y_pred
print Y_pred.max(0)
print np.mean(Y_pred, axis=0)

evaluateTraits(Y_pred, Y_test)