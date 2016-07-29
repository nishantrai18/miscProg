from sklearn.linear_model import Ridge
from reader import *
import sklearn
from sklearn.svm import SVR, LinearSVR
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor

def evaluatePred(Y_pred, Y_true):
	mmse = np.sum((Y_pred - Y_true)**2)/((Y_pred.shape[0]) * (1.0))
	bmse = Y_true.var()
	print mmse, bmse
	print ((bmse - mmse)/bmse)

if __name__ == "__main__":
	X, Y = getDataXY()
	encoder = oneHot(X)
	Xt = encoder.transform(X)
	print Xt.shape

	buySet = set()
	for i in range(X.shape[0]):
		tmpTup = (X[i][0], X[i][2])
		buySet.add(tmpTup)
	# Y_buy = [1] * Xt.shape[0]

	split = 0.75
	X_train, X_test = Xt[:(int(Xt.shape[0]*split)),:], Xt[int(Xt.shape[0]*split):, :]
	Y_train, Y_test = Y[:(int(Y.shape[0]*split)),:], Y[int(Y.shape[0]*split):, :]
	Y_train = Y_train.ravel()
	Y_test = Y_test.ravel()

	clf = Ridge(alpha = 100)
	# clf = SVR(C = 5.0, kernel = 'poly', degree = 2)
	# clf = LinearSVR(C = 100.0)
	# clf = BaggingRegressor(DecisionTreeRegressor(max_depth = 5), n_estimators = 10, n_jobs = 4)
	# clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 5), n_estimators = 100)
	# clf = DecisionTreeRegressor()
	
	clf.fit(X_train, Y_train.ravel())

	Y_pred = clf.predict(X_test)

	evaluatePred(Y_pred, Y_test)

	generatePredFile(buySet, clf, encoder)