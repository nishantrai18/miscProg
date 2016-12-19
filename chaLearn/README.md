This code is licensed under the MIT License

Code and resources for the ChaLearn Personality Trait Analysis challenge

Rough overview of the files:
----------------------------

audioPredict.py : Used for training models which use audio features. The audio features are computed using getAudioFet. 

getAudioFet.py : Extract audio features from clips. Need to enter the directories where to save and read from.

ensembleFunc.py : Functions for creating ensembles.

basicConv.py : First attempt using facial features.

fetBRegressors.py : Using facial landmarks as features. Trains models using those as features.

fetCConv.py : fetC refers to visualFetC, which refers to the face based models (After normalizing and box smoothing). This is the CNN trained over fetC.

vggMLP.py : Using background based features to predict the traits.

vggFetCConcat.py : Network for merging Facial and BG based features.

getVisualFet.py : Get Visual Features (Facial and Background)

readVideo.py : Basic video IO functions.

ensemble.py : Create ensembles and make predictions using them. Sort of the final stage in the pipeline.

regressorArmy.py : Create regressor army (Train massive number of models) and create ensembles from the less correlated ones.

dataProcess.py : Data processing helper functions.

evaluateModel.py : Evaluate the accuracy and performance of the model trained.

dlibLandmarkTest.py, dlibFaceTest.py : Face and landmark detection using Dlib

faceDetect.py : Functions for box smoothing and detecting faces.

mergeScores.py : Merging scores of a video clip. Also contains the averaging and expansion procedure.

Steps to get test predictions:
-----------------------------

Run the following command,

>> python ensemble.py

This creates a prediction (predictions.csv) file in folder code/tmpData/predictions/

You'll see the progress of the prediction. It's slightly slow (takes around 5-10 mins, since the model used is a bagged  regressor)

You might require a couple of libraries to get this to work, like dlib, sklearn, etc.
