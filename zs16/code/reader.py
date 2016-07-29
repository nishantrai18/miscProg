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

def getHospitalProfile():
	popList = {}
	with open('../resources/Dataset/HospitalProfiling.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		next(reader, None)
		for row in reader:
			fetList = []
			for i in range(3):
				tmpVal = formatField(row[i])
				if (tmpVal == "IGNORE"):
					break
				fetList.append(tmpVal)
			if (len(fetList) < 3):
				continue
			fetList = np.array(fetList)
			tmpKey = (fetList[0], fetList[1])
			if (tmpKey in popList):
				popList[tmpKey] = max(popList[tmpKey], fetList[2])
			else:
				popList[tmpKey] = fetList[2]
	csvfile.close()

	return popList

def getDataXY(currYearFlag = False, popFlag = False):

	if popFlag:
		popList = getHospitalProfile()

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

			if (popFlag):
				tmpKey = (fetList[0], fetList[2])
				if (tmpKey in popList):
					fetList[0] = popList[tmpKey]
					fetList[1] = 0
				else:
					# print tmpKey
					fetList[0] = 100
					fetList[1] = 1
					# print 'OMG'

			fetList = np.array(fetList)
			dataList.append(fetList)
	csvfile.close()
	# dataList.extend(dataList)
	# dataList.extend(dataList)
	# dataList.extend(dataList)
	dataList = np.array(dataList)

	if (currYearFlag):

		currDataList = []
		with open('../resources/Dataset/ProjectedRevenue.csv', 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter = ',')
			next(reader, None)
			for row in reader:
				fetList = []
				for i in range(1):
					tmpVal = formatField(row[i])
					if (tmpVal == "IGNORE"):
						break
					fetList.append(tmpVal)
				for i in range(1, 2):
					fetList.append(-1)
				for i in range(1, 3):
					tmpVal = formatField(row[i])
					if (tmpVal == "IGNORE"):
						break
					fetList.append(tmpVal)
				for i in range(4, 16):
					fetList.append(-1)
				for i in range(3, 4):
					tmpVal = formatField(row[i])
					if (tmpVal == "IGNORE"):
						break
					fetList.append(tmpVal)

				if (len(fetList) != 17):
					continue

				if (popFlag):
					tmpKey = (fetList[0], fetList[2])
					if (tmpKey in popList):
						fetList[0] = popList[tmpKey]
						fetList[1] = 0
					else:
						# print tmpKey
						fetList[0] = 100
						fetList[1] = 1
						# print 'OMG'

				fetList = np.array(fetList)
				currDataList.append(fetList)
		csvfile.close()
		# currDataList.extend(currDataList)
		# currDataList.extend(currDataList)
		# currDataList.extend(currDataList)
		currDataList = np.array(currDataList)

		print dataList.shape
		print currDataList.shape

		dataList = np.vstack((dataList, currDataList))

	print dataList.shape

	if (popFlag):
		X = dataList[:,[0,1,2,3]]
	else:
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

def generatePredFileC(clfBuy, clfRevenue, encBuy, encRev, scalerBuy, scalerRev, popFlag = True):
	fieldList = []
	dataList = []

	# print encoder.n_values_

	if popFlag:
		popList = getHospitalProfile()

	with open('../resources/Dataset/Solution.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		next(reader, None)
		for row in reader:
			fetList = [-1]
			for i in range(3):
				tmpVal = formatField(row[i])
				if (tmpVal == "IGNORE"):
					break
				if (tmpVal >= encRev.n_values_[i]):
					tmpVal = encRev.n_values_[i] - 1
				fetList.append(tmpVal)
			if (len(fetList) < 4):
				continue
			if (popFlag):
				tmpKey = (fetList[1], fetList[2])
				if (tmpKey in popList):
					fetList[0] = popList[tmpKey]
					fetList[1] = 0
				else:
					# print tmpKey
					fetList[0] = 100
					fetList[1] = 1

			fieldList.append(row)
			fetList = np.array(fetList)
			dataList.append(fetList)

	csvfile.close()

	X = np.array(dataList)
	print X

	if popFlag:
		Xt_rev = encRev.transform(X[:, 1:])
		Xt_rev = np.hstack((X[:,1].reshape(-1, 1), Xt_rev))
	else:
		Xt_rev = encRev.transform(X)

	if popFlag:
		Xt_buy = encBuy.transform(X[:, 1:])
		Xt_buy = np.hstack((X[:,1].reshape(-1, 1), Xt_buy))
	else:
		Xt_buy = encBuy.transform(X)

	Xt_buy = scalerBuy.transform(Xt_buy)
	Xt_rev = scalerRev.transform(Xt_rev)

	Y_pred_Revenue = clfRevenue.predict(Xt_rev)
	Y_pred_Buy = clfBuy.predict_proba(Xt_buy)

	# for t in list(Y_pred_Buy):
	# 	print t

	with open('../resources/Dataset/SolutionMine.csv', 'wb') as csvfile:
		gtwriter = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
		gtwriter.writerow(['Hospital_ID', 'District_ID', 'Instrument_ID', 'Buy_or_not', 'Revenue'])
		for i in range(0, len(fieldList)):
			buyFlag = Y_pred_Buy[i][1]
			if (buyFlag > 0.9):
				buyFlag = 1
			else:
				buyFlag = 0
			gtwriter.writerow([fieldList[i][0], fieldList[i][1], fieldList[i][2], buyFlag, int(buyFlag * int(Y_pred_Revenue[i]))])
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
