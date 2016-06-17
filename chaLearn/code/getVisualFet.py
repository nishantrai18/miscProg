from readVideo import *
import sys
import dlib

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

	videoPath = '../training/download_train-val/trainFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]

	videoPath = '../training/download_train-val/validationFiles/'
	vidNames = os.listdir(videoPath)
	vidNames = [x for x in vidNames if x.endswith(".mp4")]

	vidNames.extend(vidNamesTrain)

	# Initialize detectors, load it for face detection
	predictorPath = 'coreData/shape_predictor_68_face_landmarks.dat'
	faceDetector = dlib.get_frontal_face_detector()

	saveFetPath = 'tmpData/visualFetC/'

	if not os.path.exists(saveFetPath):
	    os.makedirs(saveFetPath)
	vidNames = vidNames

	for i in range(len(vidNames)):
		fileName = vidNames[i]
		frameList = GetFrames(videoPath+fileName, redFact = 0.5, skipLength = 5)
		
		faceList = DetectFaceInListDlib(frameList, faceDetector)
		savePath = saveFetPath + fileName.strip('.mp4')
		np.save(savePath, faceList)

		print ('\r'), ((i*(1.0))/len(vidNames)), 'part completed. Currently at file:', fileName,
		sys.stdout.flush()

	print '\n'

if __name__ == "__main__":
	getVisualFetC()