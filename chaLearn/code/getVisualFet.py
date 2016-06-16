from readVideo import *
import sys

def getVisualFetA():
	'''
	First attempt at extracting features
	Simply extract the face from each frame, followed by resizing
	The generated file for each video contains a nump array of subimages (Of faces) 
	'''

	# fileName = '../training/training_gt.csv'
	# trueMap = getTruthVal(fileName)

	print 'Started extracting features A'

	videoPath = '../training/download_train-val/trainFiles/'
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

	vidNames = vidNames[658:]

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

if __name__ == "__main__":
	getVisualFetA()