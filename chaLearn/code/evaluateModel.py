from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json

import csv
import pickle
import numpy as np
from readVideo import *
import random

def predictScore(fileName, model):
    X, _ = readData([fileName])
    X = X.reshape(X.shape[0], 3, 50, 50)
    X = X.astype('float32')
    X /= 255
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
    for i in range(len(vidNames)):
        fileName = vidNames[i]
        predVal[fileName] = predictScore(fileName, model)
        print '\r', (i*(1.0))/len(vidNames), 'part completed',
    pickle.dump(predVal, open('tmpData/predictions/predB.p', 'wb'))
    return predVal

def predictScoreList(fileName, model):
    X, _ = readData([fileName])
    X = X.reshape(X.shape[0], 3, 50, 50)
    X = X.astype('float32')
    X /= 255
    Y_pred = model.predict(X)
    return Y_pred

def predictVideos(fileList, modelName, model):
    # fileList must contain the actual paths of the files
    predVal = {}
    for i in range(len(fileList)):
        fileName = fileList[i]
        predVal[fileName] = predictScoreList(fileName, model)
        print '\r', (i*(1.0))/len(fileList), 'part completed',
    pickle.dump(predVal, open('tmpData/predictions/predList' + modelName +'_test.p', 'wb'))
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
    splitVal = 0.9
    vidNamesTest = vidNames[int(splitVal*len(vidNames))+1:]
    vidNames = vidNames[:int(splitVal*len(vidNames))]

    X_test, Y_test = readData(vidNamesTest, trueVal)
    X_test = X_test.reshape(X_test.shape[0], 3, row, col)
    X_test = X_test.astype('float32')
    X_test /= 255

    # p = pickle.load(open('tmpData/predictions/predA.p', 'rb'))
    # generatePredFile(p)
    # raw_input('FINISHED')

    modelName = 'visualFetA_BasicConv_16_32_256'
    model_file_name = 'tmpData/models/visualFetA_BasicConv_16_32_256'
    # modelName = 'visualFetA_BasicConv_Augmented_32_64_256'
    # model_file_name = 'tmpData/models/visualFetA_BasicConv_Augmented_32_64_256'

    model = model_from_json(open(model_file_name + '.json').read())
    print model_file_name
    model.load_weights(model_file_name + '_epoch_25.hdf5')
    # model.load_weights(model_file_name + '.hdf5')
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')

    print 'Model Loaded. Prediction in progress'

    # generatePredFile(evaluateValidation(model))
    predictVideos([(x) for x in vidNamesTest], modelName, model)
    # Save the prediction list in a file

    Y_pred = model.predict(X_test)

    print Y_pred
    print Y_pred.max(0)
    print np.mean(Y_pred, axis=0)

    evaluateTraits(Y_pred, Y_test)