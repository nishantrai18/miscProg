import numpy as np
from skvideo.io import VideoCapture
import cv2
import skimage
import dlib

def DetectFace(frame):
	'''
	Given a frame, detect and return the subimage containing the face
	'''
	cascPath = 'coreData/haarcascade_frontalface_default.xml'
	faceCascade = cv2.CascadeClassifier(cascPath)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.2,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	if (len(faces) != 1):
		print "Inconsistent frame. Multiple or no faces found."
		raw_input("What to do? Execution resumes after input.")

	print "Found {0} faces!".format(len(faces))

	# Draw a rectangle around the faces
	# Assuming only one face for now
	frame = np.array(frame)
	(x, y, w, h) = faces[0]
	return frame[y:(y+h),x:(x+w),:]

def DrawFace(frame):
	'''
	Given a frame, draw the box surrounding the detected faces
	Returns None
	'''

	cascPath = 'coreData/haarcascade_frontalface_default.xml'
	faceCascade = cv2.CascadeClassifier(cascPath)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.2,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	print "Found {0} faces!".format(len(faces))
	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow("Faces found", frame)
	cv2.waitKey(0)

	return None

def ViewFaceInList(frameList):
	'''
	Given a frame list, detect (track) the faces
	Returns None
	'''

	cascPath = 'coreData/haarcascade_frontalface_default.xml'
	faceCascade = cv2.CascadeClassifier(cascPath)

	for i in range(0, frameList.shape[0]):
		frame = frameList[i]
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.2,
			minNeighbors=5,
			minSize=(30, 30),
			flags = cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT
			# flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)

		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.imshow('Detected Faces', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	return None

def DetectFaceInList(frameList, faceCascade = None, debug = False):
	'''
	Given a frame list, detect (track) the faces
	Returns list of subimages containing tracekd faces
	'''

	if (faceCascade is None):
		cascPath = 'coreData/haarcascade_frontalface_default.xml'
		faceCascade = cv2.CascadeClassifier(cascPath)

	faceList = []

	for i in range(0, frameList.shape[0]):
		frame = frameList[i]
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.2,
			minNeighbors=5,
			minSize=(50, 50),
			flags = cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT
			# flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)

		if debug:
			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			cv2.imshow('Detected Faces', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		if (len(faces) == 1):
			frame = np.array(frame)
			(x, y, w, h) = faces[0]
			faceList.append(frame[y:(y+h),x:(x+w),:])

	faceList = np.array(faceList)
	return faceList

def NormalizeShape(shape, face):
	row = (face.top() + face.bottom())/2.0
	col = (face.left() + face.right())/2.0
	rowSize = np.abs(face.top() - face.bottom())
	colSize = np.abs(face.left() - face.right())
	shapeList = []
	shapeNum = 68
	for i in xrange(shapeNum):
		# Hard coded 68 value
		shapeList.append((shape.part(i).y - row)/rowSize)
		shapeList.append((shape.part(i).x - col)/colSize)
	shapeList = np.array(shapeList)
	return shapeList

def RawShape(shape, face):
	row = (face.top() + face.bottom())/2.0
	col = (face.left() + face.right())/2.0
	rowSize = np.abs(face.top() - face.bottom())
	colSize = np.abs(face.left() - face.right())
	shapeList = [row, col, rowSize, colSize]
	shapeNum = 68
	for i in xrange(shapeNum):
		# Hard coded 68 value
		shapeList.append(shape.part(i).y)
		shapeList.append(shape.part(i).x)
	shapeList = np.array(shapeList)
	return shapeList

def DetectFaceLandmarksInList(frameList, faceDetector = None, shapePredictor = None, skipLength = 2, debug = False):
	'''
	Given a frame list, detect (track) the faces
	Returns details of the facial landmarks for each frame
	'''
	if ((faceDetector is None) or (shapePredictor is None)):
		predictorPath = 'coreData/shape_predictor_68_face_landmarks.dat'
		faceDetector = dlib.get_frontal_face_detector()
		shapePredictor = dlib.shape_predictor(predictorPath)

	if (debug):
		win = dlib.image_window()
		win.clear_overlay()

	faceList = []

	for i in range(0, frameList.shape[0], skipLength):
		frame = frameList[i]
		dets = faceDetector(frame, 1)

		if debug:
			win.clear_overlay()
			win.set_image(frame)
			print("Number of faces detected: {}".format(len(dets)))
			for k, d in enumerate(dets):
				print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
				shape = shapePredictor(frame, d)
				win.add_overlay(shape)
			win.add_overlay(dets)

		dets = list(enumerate(dets))
		faceNum = len(dets)

		if (faceNum == 1):
			shape = shapePredictor(frame, dets[0][1])
			# faceShape = NormalizeShape(shape, dets[0][1])
			faceShape = RawShape(shape, dets[0][1])
			faceList.append(faceShape)

	faceList = np.array(faceList)
	return faceList

def DetectFaceInListDlib(frameList, faceDetector = None, skipLength = 2, debug = False):
	'''
	Given a frame list, detect (track) the faces
	Returns subimages of faces after normalization and smoothing the enclosing rectangle
	'''
	if ((faceDetector is None)):
		predictorPath = 'coreData/shape_predictor_68_face_landmarks.dat'
		faceDetector = dlib.get_frontal_face_detector()

	if (debug):
		win = dlib.image_window()
		win.clear_overlay()

	faceList = []
	newFrameList = []
	rowList = []
	colList = []
	detsList = []
	smoothRowSize = []
	smoothColSize = []
	winSize = (50/skipLength)

	print 'We are here'

	for i in range(0, frameList.shape[0], skipLength):
		frame = frameList[i]
		dets = faceDetector(frame, 1)
		dets = list(enumerate(dets))
		if (len(dets) != 1):
			continue
		detsList.append(dets)
		newFrameList.append(frame)
		for k, d in (dets):
			rowList.append(np.abs(d.left() - d.right()))
			colList.append(np.abs(d.top() - d.bottom()))

	print winSize
	print rowList

	for i in range(len(rowList)):
		rowAvg = np.mean(rowList[max(0,i-winSize):min(len(rowList),i+winSize)]) + 10
		colAvg = np.mean(colList[max(0,i-winSize):min(len(colList),i+winSize)]) + 10
		smoothRowSize.append(int(round(rowAvg)))
		smoothColSize.append(int(round(colAvg)))

	print smoothRowSize

	for i in range(len(detsList)):
		dets = detsList[i]
		frame = newFrameList[i]
		grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		rowc = (d.left() + d.right())/2
		colc = (d.top() + d.bottom())/2		
		rows = smoothRowSize[i]
		cols = smoothColSize[i]
		
		faceImg = grayFrame[max(0,colc-(cols/2)):min(frame.shape[0],colc+(cols/2)+1), max(0,rowc-(rows/2)):min(frame.shape[1],rowc+(rows/2)+1)]
		# Smoothing rectangle sizes (Running average of -50/+50 frames)
		faceImg = cv2.equalizeHist(faceImg)
		# Illumination (CLAHE normalization) Normalization
		faceImg = np.array(faceImg)

		if debug:
			win.clear_overlay()
			grayFrame[max(0,colc-(cols/2)):min(frame.shape[0],colc+(cols/2)+1), max(0,rowc-(rows/2)):min(frame.shape[1],rowc+(rows/2)+1)] = faceImg
			win.set_image(grayFrame)
			dlib.hit_enter_to_continue()
		
		faceList.append(faceImg)

	faceList = np.array(faceList)
	return faceList

