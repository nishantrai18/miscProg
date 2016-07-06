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


def getSortedFeatures(fetList, size = 15, sortFlag = True):
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
	if sortFlag:
		sortFet = np.sort(fetList)
	else:
		sortFet = fetList
	return sortFet

def getCompleteSortedFeatures(fetList, numTargets = 5, size = 15, sortFlag = True):

	newFet = []
	for i in range(numTargets):
		tmpFet = list(getSortedFeatures(fetList[:,i], size, sortFlag = sortFlag))
		newFet.extend(tmpFet)

	newFet = np.array(newFet)
	return newFet

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
		else:
			for i in range(5):
				Y_pred[:,i] = model[i].predict(X)
	else:
		Y_pred = model.predict(X)

	Y_pred = np.clip(Y_pred, 0, 1)
	# Y_pred[np.where(Y_pred < 0)] = 0
	# Y_pred[np.where(Y_pred > 1)] = 1

	finalScore = []
	if (merger is None):
		finalScore = np.mean(Y_pred, axis=0)
	elif (len(Y_pred) == 0):
		finalScore = np.zeros((1, 5))
		finalScore.fill(0.5)
	elif (type(merger[0]).__module__ == 'numpy'):
		for i in range(5):
			# x = getSortedFeatures(Y_pred[:,i])
			x = getCompleteSortedFeatures(Y_pred)
			finalScore.append(x.mean(0))
		finalScore = np.array(finalScore)
	else:
		for i in range(5):
			# x = getSortedFeatures(Y_pred[:,i])
			x = getCompleteSortedFeatures(Y_pred)
			y = merger[i].predict([x])[0]
			if (y < 0):
				y = 0
			if (y > 1):
				y = 1
			finalScore.append(y)
		finalScore = np.array(finalScore)
	return finalScore

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
		else:
			for i in range(5):
				Y_pred[:,i] = model[i].predict(X)
	else:
		if (X.shape[0] == 0):
			Y_pred = np.zeros((1, 5))
			Y_pred.fill(0.5)
			return Y_pred
		Y_pred = model.predict(X)

	Y_pred = np.clip(Y_pred, 0, 1)

	# Y_pred[np.where(Y_pred < 0)] = 0
	# Y_pred[np.where(Y_pred > 1)] = 1

	return Y_pred

def predictVideos(fileList, modelName, model, choice = 'A', append = 'test'):
	# fileList must contain the actual paths of the files
	predVal = {}
	for i in range(len(fileList)):
		fileName = fileList[i]
		predVal[fileName] = predictScoreList(fileName, model, choice)
		print '\r', (i*(1.0))/len(fileList), 'part completed',
		sys.stdout.flush()
	f = open('tmpData/predictions/predList' + modelName +'_' + append +'.p', 'wb')
	pickle.dump(predVal, f)
	f.close()
	return predVal

if __name__ == "__main__":
	videoPath = '../training/download_train-val/trainFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]

	fileName = '../training/training_gt.csv'
	trueVal = getTruthVal(fileName)

	for i in xrange(len(vidNames)):
		vidNames[i] = vidNames[i].strip('.mp4')

	row, col = 50, 50
	splitVal = 0.9
	vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
	vidNames = vidNames[:int(splitVal*len(vidNames))]

	# choice = 'C'
	# choice = 'B'
	choice = 'AudioA'
	# choice = 'AudioAavg'
	# action = 'genSubmit'
	action = 'getPredList'
	# action = 'getTestScore'

	# modelName = 'visualFetA_BasicConv_16_32_256'
	# model_file_name = 'tmpData/models/visualFetA_BasicConv_16_32_256'
	# modelName = 'visualFetA_BasicConv_Augmented_32_64_256'
	# model_file_name = 'tmpData/models/visualFetA_BasicConv_Augmented_32_64_256'
	# modelName = 'visualFetC_Conv_Augmented_32_64_256'
	# model_file_name = 'tmpData/models/visualFetC_Conv_Augmented_32_64_256'
	# modelName = 'visualFetF_VGG_5_128_4096_avg'
	# model_file_name = 'tmpData/models/visualFetF_VGG_5_128_4096_avg'
	# modelName = 'visualFetC_Conv_48_96_256'
	# model_file_name = 'tmpData/models/visualFetC_Conv_48_96_256'
	modelName = 'audioFetA_BAG_n50'
	model_file_name = 'tmpData/models/audioFetA_BAG_n50'
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

		if (('Conv' in modelName) or ('VGG' in modelName)):
			# X_train = X_train.reshape(X_train.shape[0], 3, row, col)
			# X_test = X_test.reshape(X_test.shape[0], 1, row, col)
			# X_test = X_test.astype('float32')
			# X_test /= 255

			model = model_from_json(open(model_file_name + '.json').read())
			print model_file_name
			# model.load_weights(model_file_name + '_epoch_25.hdf5')
			model.load_weights(model_file_name + '.hdf5')
			model.compile(loss='mean_absolute_error', optimizer='rmsprop')

		else:
			model = pickle.load(open(model_file_name + '.p', 'rb'))

		print 'Model Loaded. Prediction in progress'

		# Also change this line for test/train
		predictVideos([(x) for x in vidNames], modelName, model, choice, 'train' + str(splitVal))

	elif (action == 'getTestScore'):

		# Get the test score for the whole list of data points

		X_test, Y_test = readData(vidNamesTest, trueVal, choice)
		# X_test = X_test.reshape(X_test.shape[0], 3, row, col)
		# X_test = X_test.reshape(X_test.shape[0], 1, row, col)
		# X_test = X_test.astype('float32')
		# X_test /= 255

		if (('Conv' in modelName) or ('VGG' in modelName)):
			model = model_from_json(open(model_file_name + '.json').read())
			print model_file_name
			# model.load_weights(model_file_name + '_epoch_25.hdf5')
			model.load_weights(model_file_name + '.hdf5')
			model.compile(loss='mean_absolute_error', optimizer='rmsprop')

		print 'Model Loaded. Prediction in progress'

		Y_pred = model.predict(X_test)
		evaluateTraits(Y_pred, Y_test)

	elif (action == 'genSubmit'):
		mergeName = '_LS'
		# merger = pickle.load(open('tmpData/models/mergeScore_FetC_LS.p', 'rb'))
		# merger = pickle.load(open('tmpData/models/mergeScore_FetAudioA_BAG_WGT.p', 'rb'))
		merger = pickle.load(open('tmpData/models/mergeScore_FetC_48_96_LS.p', 'rb'))

		if (('Conv' in modelName) or ('VGG' in modelName)):

			model = model_from_json(open(model_file_name + '.json').read())
			print model_file_name
			# model.load_weights(model_file_name + '_epoch_25.hdf5')
			model.load_weights(model_file_name + '.hdf5')
			model.compile(loss='mean_absolute_error', optimizer='rmsprop')

		else:
			model = pickle.load(open(model_file_name + '.p', 'rb'))
			# print 'Model Loaded. Prediction in progress'
		
		print 'Model Loaded. Prediction in progress'
		generatePredFile(evaluateValidation(model, merger, modelName + mergeName, choice))
		# f = open('tmpData/predictions/valPredictionaudioFetA_BAG_n50_LS.p', 'rb')
		# generatePredFile(pickle.load(f))
		# f.close()
	
	# p = pickle.load(open('tmpData/predictions/predA.p', 'rb'))
	# generatePredFile(p)
	# raw_input('FINISHED')