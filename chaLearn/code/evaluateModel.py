from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json

import numpy as np
from readVideo import *
import random

def evaluateTraits(p, gt):
    if (len(p) == len(gt)):
        for i in range(len(p)):
            if (len(p[i]) != 5) or (len(gt[i]) != 5):
                print "Inputs must be a list of 5 values within the range [0,1]. Traits could not be evaluated."
                return
            for j in range(len(p[i])):
                if p[i][j] < 0 or p[i][j] > 1 or gt[i][j] < 0 or gt[i][j] > 1:
                    print "Inputs must be values in the range [0,1]. Traits could not be evaluated."
                    return
    
    errors = np.abs(p-gt)
    meanAccs = 1-np.mean(errors, axis=0)
    
    print "\nAverage accuracy of "+str(np.mean(meanAccs))+": "
    
    # These scores are reported.
    print "Accuracy predicting Extraversion: "+str(meanAccs[0])
    print "Accuracy predicting Agreeableness: "+str(meanAccs[1])
    print "Accuracy predicting Conscientiousness: "+str(meanAccs[2])
    print "Accuracy predicting Neuroticism: "+str(meanAccs[3])
    print "Accuracy predicting Openness to Experience: "+str(meanAccs[4])
    print "\n"
        
    return meanAccs

def predictScore(fileName, model):
    X, _ = readData([fileName])
    Y_pred = model.predict(X)
    finalScore = np.mean(Y_pred, axis=0)
    return finalScore

def evaluateValidation(model):
    predVal = {}
    videoPath = '../training/download_train-val/validationFiles/'
    vidNames = os.listdir(videoPath)
    vidNames = [x for x in vidNames if x.endswith(".mp4")]
    for i in xrange(len(vidNames)):
        vidNames[i] = vidNames[i].strip('.mp4')
    for fileName in vidNames:
        predVal[fileName] = predictScore(fileName, model)
    return predVal

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

X_test, Y_test = readData(vidNamesTest, trueVal)
X_test = X_test.reshape(X_test.shape[0], 3, row, col)
X_test = X_test.astype('float32')
X_test /= 255

model_file_name = 'tmpData/models/visualFetA_BasicConv_16_32_256'
# model_file_name = 'tmpData/models/visualFetA_BasicConv_Augmented_32_64_256'

model = model_from_json(open(model_file_name + '.json').read())
print model_file_name
model.load_weights(model_file_name + '_epoch_25.hdf5')
# model.load_weights(model_file_name + '.hdf5')
model.compile(loss='mean_absolute_error', optimizer='rmsprop')

print 'Model Loaded. Prediction in progress'

Y_pred = model.predict(X_test)

print Y_pred
print Y_pred.max(0)
print np.mean(Y_pred, axis=0)

evaluateTraits(Y_pred, Y_test)