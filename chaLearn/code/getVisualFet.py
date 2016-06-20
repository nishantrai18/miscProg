from readVideo import *
import sys
import dlib
from keras.optimizers import SGD

def getVisualFetA():
	'''
	First attempt at extracting features
	Simply extract the face from each frame, followed by resizing
	The generated file for each video contains a nump array of subimages (Of faces) 
	'''

	# fileName = '../training/training_gt.csv'
	# trueMap = getTruthVal(fileName)

	print 'Started extracting features A'

	# videoPath = '../training/download_train-val/trainFiles/'
	videoPath = '../training/download_train-val/validationFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]

	# Initialize cascade, load it for face detection
	cascPath = 'coreData/haarcascade_frontalface_default.xml'
	faceCascade = cv2.CascadeClassifier(cascPath)

	saveFetPath = 'tmpData/visualFetA/'
	saveVidPath = 'tmpData/vidData/'

	if not os.path.exists(saveFetPath):
		os.makedirs(saveFetPath)

	if not os.path.exists(saveVidPath):
		os.makedirs(saveVidPath)

	vidNames = vidNames

	for i in range(len(vidNames)):
		fileName = vidNames[i]
		frameList = GetFrames(videoPath+fileName, redFact = 0.5, skipLength = 5)
		savePath = saveVidPath + fileName.strip('.mp4')
		
		# np.save(savePath, frameList)
		# Do not save, too large!

		faceList = DetectFaceInList(frameList, faceCascade)
		faceList = equalizeImgList(faceList)
		savePath = saveFetPath + fileName.strip('.mp4')
		np.save(savePath, faceList)

		print ('\r'), ((i*(1.0))/len(vidNames)), 'part completed. Currently at file:', fileName,
		sys.stdout.flush()

	print '\n'

def getVisualFetB():
	'''
	Second attempt at extracting features
	Simply extract the face from each frame, followed by extracting details of the landmarks
	The generated file for each video contains a numpy array of the vectors (Of facial landmarks) 
	'''

	# fileName = '../training/training_gt.csv'
	# trueMap = getTruthVal(fileName)

	print 'Started extracting features B'

	videoPath = '../training/download_train-val/trainFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]

	# videoPath = '../training/download_train-val/validationFiles/'
	# vidNames = os.listdir(videoPath)
	# vidNames = [x for x in vidNames if x.endswith(".mp4")]

	# vidNames.extend(vidNamesTrain)

	# Initialize detectors, load it for face detection
	predictorPath = 'coreData/shape_predictor_68_face_landmarks.dat'
	faceDetector = dlib.get_frontal_face_detector()
	shapePredictor = dlib.shape_predictor(predictorPath)

	saveFetPath = 'tmpData/visualFetB/'
	saveVidPath = 'tmpData/vidData/'

	if not os.path.exists(saveFetPath):
		os.makedirs(saveFetPath)

	if not os.path.exists(saveVidPath):
		os.makedirs(saveVidPath)

	vidNames = vidNames

	for i in range(len(vidNames)):
		fileName = vidNames[i]
		frameList = GetFrames(videoPath+fileName, redFact = 0.5, skipLength = 5)
		savePath = saveVidPath + fileName.strip('.mp4')
		
		# np.save(savePath, frameList)
		# Do not save, too large!

		faceList = DetectFaceLandmarksInList(frameList, faceDetector, shapePredictor)
		savePath = saveFetPath + fileName.strip('.mp4')
		np.save(savePath, faceList)

		print ('\r'), ((i*(1.0))/len(vidNames)), 'part completed. Currently at file:', fileName,
		sys.stdout.flush()

	print '\n'

def getVisualFetC():
	'''
	Third attempt at extracting features
	Extract the face from each frame using the dlib face detector, followed by normalization
	Box size normalization is also performed along with taking a larger crop for allowing more augmentation
	The generated file for each video contains a numpy array of faces
	'''

	# fileName = '../training/training_gt.csv'
	# trueMap = getTruthVal(fileName)

	print 'Started extracting features C'

	videoPath = '../training/download_train-val/validationFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]

	# videoPath = '../training/download_train-val/trainFiles/'
	# vidNames = os.listdir(videoPath)
	# vidNames = [x for x in vidNames if x.endswith(".mp4")]

	# vidNames.extend(vidNamesTest)

	# Initialize detectors, load it for face detection
	predictorPath = 'coreData/shape_predictor_68_face_landmarks.dat'
	faceDetector = dlib.get_frontal_face_detector()

	saveFetPath = 'tmpData/visualFetC/'

	if not os.path.exists(saveFetPath):
		os.makedirs(saveFetPath)
	vidNames = vidNames

	for i in range(len(vidNames)):
		fileName = vidNames[i]

		if (os.path.isfile(saveFetPath+fileName.strip('.mp4')+'.npy')):
			continue

		frameList = GetFrames(videoPath+fileName, redFact = 0.5, skipLength = 5)
		
		faceList = DetectFaceInListDlib(frameList, faceDetector)
		savePath = saveFetPath + fileName.strip('.mp4')
		np.save(savePath, faceList)

		print ('\r'), ((i*(1.0))/len(vidNames)), 'part completed. Currently at file:', fileName,
		sys.stdout.flush()

	print '\n'

def VGG_16():
	weights_path = '/home/nishant/Documents/Academics/CS676A/project/cs676project/code/models2/CNN/vgg16_weights.h5'

	from keras.models import Sequential
	from keras.layers.core import Flatten, Dense, Dropout
	from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)

	#Remove the last two layers to get the 4096D activations
	model.layers.pop()
	model.layers.pop()

	# Fix required for the output to be 4096D in newer versions of keras
	model.outputs = [model.layers[-1].output]
	model.layers[-1].outbound_nodes = []

	print 'VGG Model loading complete!'

	return model

def getVisualFetF():
	'''
	Random crops taken from a frame. Passed through VGG to get a 4096D embedding.
	Then max (or average) pooled. Used as features for further use.
	The generated file for each video contains a numpy array of faces
	'''

	# fileName = '../training/training_gt.csv'
	# trueMap = getTruthVal(fileName)

	print 'Started extracting features F'

	videoPath = '../training/download_train-val/validationFiles/'
	vidNamesTest = os.listdir(videoPath)
	vidNamesTest = [x for x in vidNamesTest if x.endswith(".mp4")]

	videoPath = '../training/download_train-val/trainFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]

	vidNames.extend(vidNamesTest)

	saveFetPath = 'tmpData/visualFetF/'

	vggModel = VGG_16()
	# this is standard VGG 16 without the last two layers
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	vggModel.compile(optimizer=sgd, loss='categorical_crossentropy')

	if not os.path.exists(saveFetPath):
		os.makedirs(saveFetPath)
	vidNames = vidNames

	for i in range(len(vidNames)):
		fileName = vidNames[i]

		if (os.path.isfile(saveFetPath+fileName.strip('.mp4')+'.npy')):
			continue

		frameList = GetFrames(videoPath+fileName, redFact = 0.5, skipLength = 160)
		
		fetList = GetBGFeatures(frameList, vggModel, numCrops = 16)
		# Returns a set of features for each video. The desired output from it should be the scores.

		savePath = saveFetPath + fileName.strip('.mp4')
		np.save(savePath, fetList)

		print ('\r'), ((i*(1.0))/len(vidNames)), 'part completed. Currently at file:', fileName,
		sys.stdout.flush()

	print '\n'

if __name__ == "__main__":
	getVisualFetF()