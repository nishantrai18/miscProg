from keras.models import Sequential, Graph
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json

import cPickle as pickle
import numpy as np
from readVideo import *
import random

from sklearn import preprocessing
from sklearn.externals import joblib

def vggFetF():
	model_file_name = 'tmpData/models/visualFetF_VGGExtSamples_5_128_4096_avg'

	model = Sequential()
	# # input: 4096 dimension vectors.
	model.add(Dense(128, input_dim = 4096, init='uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(5))
	model.add(Activation('sigmoid'))

	print model_file_name

	model.load_weights(model_file_name + '.hdf5')

	#Remove the last layers to get the 128D representations
	model.layers.pop()
	model.layers.pop()

	# print model.layers
	# Freeze model
	for layer in model.layers:
	    layer.trainable = False

	# Fix required for the output to be 4096D in newer versions of keras
	# model.outputs = [model.layers[-1].output]
	# model.layers[-1].outbound_nodes = []

	print 'vggFetF Model loading complete!'

	return model

def convFetC():
	model_file_name = 'tmpData/models/visualFetC_Conv_Augmented_32_64_256'

	print model_file_name

	newModel = model_from_json(open(model_file_name + '.json').read())
	newModel.load_weights(model_file_name + '.hdf5')

	#Remove the last layers to get the 256D representations
	newModel.layers.pop()
	newModel.layers.pop()

	# print newModel.layers
	# Freeze model
	for layer in newModel.layers:
	    layer.trainable = False

	# Fix required for the output to be 4096D in newer versions of keras
	# newModel.outputs = [newModel.layers[-1].output]
	# newModel.layers[-1].outbound_nodes = []

	print 'convFetC Model loading complete!'

	return newModel

videoPath = '../training/download_train-val/trainFiles/'
vidNames = os.listdir(videoPath)
vidNames = [x for x in vidNames if x.endswith(".mp4")]

fileName = '../training/training_gt.csv'
trueVal = getTruthVal(fileName)

for i in xrange(len(vidNames)):
	vidNames[i] = vidNames[i].strip('.mp4')

poolType = 'avg'
row, col = 50, 50
splitVal = 0.9
vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
vidNames = vidNames[:int(splitVal*len(vidNames))]

X_test_conv, X_test_vgg, Y_test = readData(vidNamesTest, trueVal, 'CF', poolType = poolType)
X_test_conv = X_test_conv.reshape(X_test_conv.shape[0], 1, row, col)
X_test_conv = X_test_conv.astype('float32')
X_test_conv /= 255

numPerBatch = 50
numBatch = (len(vidNames)/numPerBatch)

model_save_interval = 10
num_epochs = 50

model_file_name = 'tmpData/models/VGG_FetC_Concat_256_128_' + poolType

vggModel = vggFetF()
convModel = convFetC()

print vggModel.input_shape
print vggModel.output_shape

print convModel.input_shape
print convModel.output_shape

tmpmodel = Sequential()
# # input: 4096 dimension vectors.
tmpmodel.add(Dense(128, input_dim = 4096, init='uniform'))
tmpmodel.add(Activation('relu'))
tmpmodel.add(Dropout(0.25))

model = Graph()
model.add_input(name='convInput', input_shape = (1, row, col))
model.add_input(name='vggInput', input_shape = (4096,))
model.add_node(convModel, name='conv', input='convInput')
model.add_node(vggModel, name='vgg', input='vggInput')

# Can add another hidden layer in between
model.add_node(Dense(5, init='uniform', activation = 'sigmoid'), name='final', inputs=['conv', 'vgg'], merge_mode='concat')
model.add_output(name='output', input='final')

model.compile(loss = {'output': 'mean_absolute_error'}, optimizer = 'rmsprop')
# Prefer mean_absolute_error, since that is the one used for the challenge

# # Saving model
jsonString = model.to_json()
open(model_file_name + '.json', 'w').write(jsonString)

# model = model_from_json(open(model_file_name + '.json').read())
# print model_file_name
# # model.load_weights(model_file_name + '_epoch_25.hdf5')
# model.load_weights(model_file_name + '.hdf5')

minScore = 1.0

print 'Training started...'
for k in range(num_epochs):
	#shuffle the data points before going through them
	random.shuffle(vidNames)

	progbar = generic_utils.Progbar(len(vidNames))

	for i in xrange(numBatch):
		# Read numPerBatch files, get the images and answers

		# print 'Starting reading the batch'
		X_batch_conv, X_batch_vgg, Y_batch = readData(vidNames[(i*numPerBatch):((i+1)*numPerBatch)], trueVal, 'CF', poolType = poolType)
		X_batch_conv = X_batch_conv.reshape(X_batch_conv.shape[0], 1, row, col)
		X_batch_conv = X_batch_conv.astype('float32')
		X_batch_conv /= 255

		# print 'Finished reading the batch'

		# print 'Training on Batch'
		loss = model.train_on_batch({'convInput': X_batch_conv, 'vggInput': X_batch_vgg, 'output': Y_batch})
		# Train the model
		# print 'Finished training on Batch'

		progbar.add(numPerBatch, values=[("train loss", loss)])

	#print type(loss)
	if k%model_save_interval == 0:
		model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))

	score = model.evaluate({'convInput': X_test_conv, 'vggInput': X_test_vgg, 'output': Y_test}, verbose=0)

	print "For epoch", k, ",Testing loss is", score
	if minScore > score:
		model.save_weights(model_file_name + '.hdf5', overwrite = True)
		minScore = score