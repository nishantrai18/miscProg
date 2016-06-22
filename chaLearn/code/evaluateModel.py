from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json

import csv
import cPickle as pickle
import numpy as np
from readVideo import *
import random, sys

def predictScore(fileName, model, merger = None, choice = 'A'):
	X, _ = readData([fileName], None, choice)
	if (choice == 'A'):
		X = X.reshape(X.shape[0], 3, 50, 50)
		X = X.astype('float32')
		X /= 255
	elif (choice == 'C'):
		X = X.reshape(X.shape[0], 1, 50, 50)
		X = X.astype('float32')
		X /= 255

	if (isinstance(model, list)):
		Y_pred = np.zeros((X.shape[0], 5))
		if (X.shape[0] == 0):
			Y_pred = np.zeros((1, 5))
			Y_pred.fill(0.5)
			# print Y_pred
			return Y_pred
		for i in range(5):
			Y_pred[:,i] = model[i].predict(X)
	else:
		Y_pred = model.predict(X)

	finalScore = []
	if (merger is None):
		finalScore = np.mean(Y_pred, axis=0)
	elif (len(Y_pred) == 0):
		finalScore = np.array([0.5]*5)
	elif (type(merger[0]).__module__ == 'numpy'):
		for i in range(5):
			x = getSortedFeatures(Y_pred[:,i])
			finalScore.append(x.mean(0))
		finalScore = np.array(finalScore)
	else:
		for i in range(5):
			x = getSortedFeatures(Y_pred[:,i])
			y = merger[i].predict([x])[0]
			finalScore.append(y)
		finalScore = np.array(finalScore)
	return finalScore

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

def evaluateValidation(model, merger = None, modelName = '', choice = 'A'):
	predVal = {}
	videoPath = '../training/download_train-val/validationFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]
	for i in xrange(len(vidNames)):
		vidNames[i] = vidNames[i].strip('.mp4')
	for i in range(len(vidNames)):
		fileName = vidNames[i]
		predVal[fileName] = predictScore(fileName, model, merger, choice)
		print '\r', (i*(1.0))/len(vidNames), 'part completed',
		sys.stdout.flush()
	pickle.dump(predVal, open('tmpData/predictions/valPrediction' + modelName + '.p', 'wb'))
	return predVal

def predictScoreList(fileName, model, choice = 'A'):
	X, _ = readData([fileName], None, choice)
	if (choice == 'A'):
		X = X.reshape(X.shape[0], 3, 50, 50)
		X /= 255
		X = X.astype('float32')
	elif (choice == 'C'):
		X = X.reshape(X.shape[0], 1, 50, 50)
		X /= 255
		X = X.astype('float32')

	if (isinstance(model, list)):
		Y_pred = np.zeros((X.shape[0], 5))
		if (X.shape[0] == 0):
			Y_pred = np.zeros((1, 5))
			Y_pred.fill(0.5)
			# print Y_pred
			return Y_pred
		for i in range(5):
			Y_pred[:,i] = model[i].predict(X)
	else:
		Y_pred = model.predict(X)

	return Y_pred

def predictVideos(fileList, modelName, model, choice = 'A', append = 'test'):
	# fileList must contain the actual paths of the files
	predVal = {}
	for i in range(len(fileList)):
		fileName = fileList[i]
		predVal[fileName] = predictScoreList(fileName, model, choice)
		print '\r', (i*(1.0))/len(fileList), 'part completed',
		sys.stdout.flush()
	pickle.dump(predVal, open('tmpData/predictions/predList' + modelName +'_' + append +'.p', 'wb'))
	return predVal

def generatePredFile(p, subset='validation'):
	vnames = []
	with open('../training/'+subset+'_gt.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		next(reader, None)
		for row in reader:
			vnames.append(row[0])
	csvfile.close()
	with open('tmpData/predictions/predictions.csv', 'wb') as csvfile:
		gtwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		gtwriter.writerow(['VideoName', 'ValueExtraversion', 'ValueAgreeableness', 'ValueConscientiousness', 'ValueNeurotisicm','ValueOpenness'])
		for i in range(0,len(vnames)):
			vnames[i] = vnames[i].strip('.mp4')
			if (isinstance(p[vnames[i]], np.float64)):
				p[vnames[i]] = [0.5]*5
			gtwriter.writerow([vnames[i]+'.mp4', p[vnames[i]][0], p[vnames[i]][1], p[vnames[i]][2], p[vnames[i]][3], p[vnames[i]][4]])
	csvfile.close()

if __name__ == "__main__":
	videoPath = '../training/download_train-val/trainFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]

	fileName = '../training/training_gt.csv'
	trueVal = getTruthVal(fileName)

	for i in xrange(len(vidNames)):
		vidNames[i] = vidNames[i].strip('.mp4')

	row, col = 50, 50
	splitVal = 0.4
	vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
	vidNames = vidNames[:int(splitVal*len(vidNames))]

	choice = 'C'
	# choice = 'B'
	# choice = 'AudioA'
	action = 'genSubmit'
	# action = 'getPredList'

	# modelName = 'visualFetA_BasicConv_16_32_256'
	# model_file_name = 'tmpData/models/visualFetA_BasicConv_16_32_256'
	# modelName = 'visualFetA_BasicConv_Augmented_32_64_256'
	# model_file_name = 'tmpData/models/visualFetA_BasicConv_Augmented_32_64_256'
	modelName = 'visualFetC_Conv_Augmented_32_64_256'
	model_file_name = 'tmpData/models/visualFetC_Conv_Augmented_32_64_256'
	# modelName = 'audioFetA_BAG_n50'
	# model_file_name = 'tmpData/models/audioFetA_BAG_n50'
	# modelName = 'visualFetB_MISC'
	# model_file_name = 'tmpData/models/visualFetB_MISC'

	if (action == 'getPredList'):

		# Get prediction list for each video in training or testing data
		# Can be used for merging scores later

		# Change the five lines below to get test or train prediction lists
		# X_train, Y_train = readData(vidNames, trueVal, choice)
		# # X_train = X_train.reshape(X_train.shape[0], 3, row, col)
		# X_train = X_train.reshape(X_train.shape[0], 1, row, col)
		# X_train = X_train.astype('float32')
		# X_train /= 255

		# X_test, Y_test = readData(vidNamesTest, trueVal, choice)

		if ('Conv' in modelName):
			# X_train = X_train.reshape(X_train.shape[0], 3, row, col)
			X_test = X_test.reshape(X_test.shape[0], 1, row, col)
			X_test = X_test.astype('float32')
			X_test /= 255

			model = model_from_json(open(model_file_name + '.json').read())
			print model_file_name
			# model.load_weights(model_file_name + '_epoch_25.hdf5')
			model.load_weights(model_file_name + '.hdf5')
			model.compile(loss='mean_absolute_error', optimizer='rmsprop')

		else:
			model = pickle.load(open(model_file_name + '.p', 'rb'))

		print 'Model Loaded. Prediction in progress'

		# Also change this line for test/train
		predictVideos([(x) for x in vidNamesTest], modelName, model, choice, 'test' + str(splitVal))

	elif (action == 'getTestScore'):

		# Get the test score for the whole list of data points

		X_test, Y_test = readData(vidNamesTest, trueVal, 'C')
		# X_test = X_test.reshape(X_test.shape[0], 3, row, col)
		X_test = X_test.reshape(X_test.shape[0], 1, row, col)
		X_test = X_test.astype('float32')
		X_test /= 255

		model = model_from_json(open(model_file_name + '.json').read())
		print model_file_name
		# model.load_weights(model_file_name + '_epoch_25.hdf5')
		model.load_weights(model_file_name + '.hdf5')
		model.compile(loss='mean_absolute_error', optimizer='rmsprop')

		print 'Model Loaded. Prediction in progress'

		Y_pred = model.predict(X_test)
		evaluateTraits(Y_pred, Y_test)

	elif (action == 'genSubmit'):
		mergeName = '_WGT'
		merger = pickle.load(open('tmpData/models/mergeScore_FetC_WGT.p', 'rb'))

		if ('Conv' in modelName):

			model = model_from_json(open(model_file_name + '.json').read())
			print model_file_name
			# model.load_weights(model_file_name + '_epoch_25.hdf5')
			model.load_weights(model_file_name + '.hdf5')
			model.compile(loss='mean_absolute_error', optimizer='rmsprop')

		else:
			model = pickle.load(open(model_file_name + '.p', 'rb'))

		print 'Model Loaded. Prediction in progress'
		generatePredFile(evaluateValidation(model, merger, modelName + mergeName, 'C'))
	
	# p = pickle.load(open('tmpData/predictions/predA.p', 'rb'))
	# generatePredFile(p)
	# raw_input('FINISHED')