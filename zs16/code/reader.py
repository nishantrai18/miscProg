import csv
import cPickle as pickle
import numpy as np
import random, sys
from sklearn.preprocessing import OneHotEncoder

def getInt(val):
	try:
		a = int(val)
	except ValueError:
		return "IGNORE"
	return a

def formatField(val):
	if ('Hospital ' in val):
		val = val.strip('Hospital ')
	elif ('Region ' in val):
		val = val.strip('Region ')
	elif ('District ' in val):
		val = val.strip('District ')
	elif ('Instrument ' in val):
		val = val.strip('Instrument ')
	return getInt(val)

def getDataXY():
	dataList = []
	with open('../resources/Dataset/HospitalRevenue.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		next(reader, None)
		for row in reader:
			fetList = []
			for i in range(17):
				tmpVal = formatField(row[i])
				if (tmpVal == "IGNORE"):
					break
				fetList.append(tmpVal)
			if (len(fetList) < 16):
				continue
			fetList = np.array(fetList)
			dataList.append(fetList)
	csvfile.close()
	dataList = np.array(dataList)

	print dataList.shape

	X = dataList[:,[0,2,3]]
	# Change this to modify the features chosen
	Y = dataList[:,-1:]

	print X.shape
	print Y.shape

	return X, Y

def generatePredFile(buySet, clf, encoder):
	fieldList = []
	dataList = []

	print encoder.n_values_

	with open('../resources/Dataset/Solution.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		next(reader, None)
		for row in reader:
			fetList = []
			for i in range(3):
				tmpVal = formatField(row[i])
				if (tmpVal == "IGNORE"):
					break
				if (tmpVal >= encoder.n_values_[i]):
					tmpVal = encoder.n_values_[i] - 1
				fetList.append(tmpVal)
			if (len(fetList) < 3):
				continue
			fieldList.append(row)
			fetList = np.array(fetList)
			dataList.append(fetList)
	csvfile.close()

	X = np.array(dataList)
	print X
	Xt = encoder.transform(X)
	Y_pred = clf.predict(Xt)
	# Y_pred_Revenue = clfRevenue.predict(Xt)
	# Y_pred_Buy = clfBuy.predict(Xt)

	with open('../resources/Dataset/SolutionMine.csv', 'wb') as csvfile:
		gtwriter = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
		gtwriter.writerow(['Hospital_ID', 'District_ID', 'Instrument_ID', 'Buy_or_not', 'Revenue'])
		for i in range(0, len(fieldList)):
			tmpTup = (X[i][0], X[i][2])
			buyFlag = 0
			# print tmpTup
			if (tmpTup in buySet):
				# print "GOOD"
				buyFlag = 1
			gtwriter.writerow([fieldList[i][0], fieldList[i][1], fieldList[i][2], buyFlag, buyFlag * int(Y_pred[i])])
	csvfile.close()

def oneHot(X):
	encoder = OneHotEncoder(sparse = False)
	encoder.fit(X)
	return encoder

if __name__ == "__main__":
	X, Y = getDataXY()
	encoder = oneHot(X)
	Xt = encoder.transform(X)
	print Xt.shape
