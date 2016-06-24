from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.layers.normalization import LRN2D

import numpy as np
from readVideo import *
import random

from sklearn import preprocessing
from sklearn.externals import joblib

videoPath = '../training/download_train-val/trainFiles/'
vidNames = os.listdir(videoPath)
vidNames = [x for x in vidNames if x.endswith(".mp4")]

fileName = '../training/training_gt.csv'
trueVal = getTruthVal(fileName)

for i in xrange(len(vidNames)):
	vidNames[i] = vidNames[i].strip('.mp4')
# Contains the list of videos, been stripped of .mp4

# Training and testing splits
row, col = 50, 50
splitVal = 0.9
vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
vidNames = vidNames[:int(splitVal*len(vidNames))]

X_test, Y_test = readData(vidNamesTest, trueVal, 'C')
X_test = X_test.reshape(X_test.shape[0], 1, row, col)
X_test = X_test.astype('float32')
X_test /= 255

numPerBatch = 50
numBatch = (len(vidNames)/numPerBatch)

model_save_interval = 5
num_epochs = 100

model_file_name = 'tmpData/models/visualFetC_ConvLRN_48_96_256'

model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 64 convolution filters of size 3x3 each.

model.add(Convolution2D(48, 3, 3, border_mode='valid', input_shape=(1, row, col)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LRN2D())
model.add(Dropout(0.5))

model.add(Convolution2D(96, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(LRN2D())
model.add(Dropout(0.5))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(loss='mean_absolute_error', optimizer='rmsprop')
# Prefer mean_absolute_error, since that is the one used for the challenge

# Saving model
jsonString = model.to_json()
open(model_file_name + '.json', 'w').write(jsonString)

minScore = 1.0

# raw_input('WAIT')

print 'Training started...'
for k in xrange(num_epochs):
	#shuffle the data points before going through them
	random.shuffle(vidNames)

	progbar = generic_utils.Progbar(len(vidNames))

	for i in xrange(numBatch):
		# Read numPerBatch files, get the images and answers

		# print 'Starting reading the batch'

		X_batch, Y_batch = readData(vidNames[(i*numPerBatch):((i+1)*numPerBatch)], trueVal, 'C')
		# print X_batch.shape
		X_batch = X_batch.reshape(X_batch.shape[0], 1, row, col)
		# print X_batch.shape

		# print 'Finished reading the batch'

		X_batch = X_batch.astype('float32')
		X_batch /= 255
		# Augment the data (Currently 15000 images per batch, try for 60000)

		# print 'Training on Batch'
		loss = model.train_on_batch(X_batch, Y_batch)
		# Train the model
		# print 'Finished training on Batch'

		progbar.add(numPerBatch, values=[("train loss", loss)])

	#print type(loss)
	if k%model_save_interval == 0:
		model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))

	score = model.evaluate(X_test, Y_test, verbose=0)
	print "For epoch", k, ",Testing loss is", score
	if minScore > score:
		model.save_weights(model_file_name + '.hdf5', overwrite = True)
		minScore = score