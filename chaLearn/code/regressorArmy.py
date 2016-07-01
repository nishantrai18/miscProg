import cPickle as pickle
import numpy as np

import random
import sklearn
from sklearn.svm import SVR, LinearSVR
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

def createRegressorArmy(regList, dataList, numLimit = 50, accThreshold = 0.88, scaleFlag = False):
	'''
	Input: regList, List of possible regressor types (List of strings)
		   dataList, List of (Training, Testing) data pairs
		   			 (0.5, 0.5 split recommended)
		   numLimit, Number limit for each type of classifier
		   numPerParam, Number of regressors trained per hyperparameter
		   bagVal, Bagging threshold for training multiple regressors
		   accThreshold, Accuracy/Score Threshold for models
		   scaleFlag, Scaling/Not Scaling flag, adds variety to the createClassifierArmy
	
	Output: List of classifiers
			List containing history of training

	Side effects: Write output to a file (folder in ensemble)
				  (Optional) Create list of dictionaries with the
				  predictions for another set (Ensure that the 
				  required data is given in the same order as dataList)
	'''

	'''
	Steps Involved:

		Consists of a set of classifiers already present
		Go through a loop to check if it is present in the provided arguments
		The inside statements consists of case statements
		Vary hyperparameters smoothly (mostly hardcoded), check and break on the basis of accThreshold
		For each hyperparameter, train numPerParam regressors via bagging (Around 0.75 (bagging Parameter)).
	'''
