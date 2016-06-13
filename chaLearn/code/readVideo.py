import numpy as np
from skvideo.io import VideoCapture
import cv2
import skimage
import os

def PlayVideo(fileName, redFact = 0.5):
    '''
    Plays video using opencv functions
    Press 'q' to stop in between
    returns None
    '''
    cap = VideoCapture(fileName)
    cap.open()
    while True:
        retval, image = cap.read()
        print len(image), len(image[0])
        if not retval:
            break
        image = cv2.resize(image, None, fx=redFact, fy=redFact)
        image = image[:,:,::-1]
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def GetFrames(fileName, redFact = 0.5, skipLength = 1):
    '''
    returns numpy array of frames
    '''
    cap = VideoCapture(fileName)
    cap.open()

    retval, image = cap.read()  
    # Read first to get an estimate of the number of frames

    frameList = []
    cnt = 0

    print "Started creating Frame List"

    while True:
        if not retval:
            break
        image = cv2.resize(image, None, fx=redFact, fy=redFact)
        image = image[:,:,::-1]
        image = np.array(image, dtype = np.uint8)
        if (cnt == 0):
            frameList.append(image)
        cnt = (cnt+1)%skipLength
        retval, image = cap.read()
    cap.release()

    print "Finished creating Frame List"

    print len(frameList)
    frameList = np.array(frameList)
    print frameList.shape
    return frameList

if __name__ == "__main__":
    videoPath = '../training/download_train-val/trainFiles/'
    vidNames = os.listdir(videoPath)
    vidNames = [x for x in vidNames if x.endswith(".mp4")]
    fileName = vidNames[0]

    # PlayVideo(videoPath+fileName)
    GetFrames(videoPath+fileName)