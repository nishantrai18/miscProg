from readVideo import *

def getVisualFetA():
	'''
	First attempt at extracting features
	Simply extract the face from each frame, followed by resizing
	The generated file for each video contains a nump array of subimages (Of faces) 
	'''

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

	for fileName in vidNames:
		frameList = GetFrames(videoPath+fileName, redFact = 0.5, skipLength = 5)
		savePath = saveVidPath + fileName.strip('.mp4')
		print savePath
		raw_input('Alright?')
		np.save(savePath, frameList)
		faceList = DetectFaceInList(frameList, faceCascade)
		faceList = equalizeImgList(faceList)
		savePath = saveFetPath + fileName.strip('.mp4')
		print savePath
		raw_input('Alright?')
		np.save(savePath, faceList)

if __name__ == "__main__":
	getVisualFetA()