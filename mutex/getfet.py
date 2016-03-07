import pandas
from math import isnan
from random import shuffle
import numpy as np
from sklearn import preprocessing
import pickle

keyStat = {}
idDict = {}
ids = 0
idsList = {}
le = {}

for i in range(0,100):
	le[i] = None
	idsList[i] = 0

def isfloat(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

def GetID(val):
	global idDict
	global ids
	if (val != val):
		return -1
	if (val not in idDict):
		idDict[val] = ids
		ids +=1
	return idDict[val]

def GetCustomID(dic, key):
	if (key != key):
		return -1
	if (key not in dic):
		return -1
	return dic[key]

def clean_list(lis, t):
	ans = []
	tot = 0
	cnt = 1
	for i in range(0,len(lis)):
		tmp = GetID(lis[i])
		if (not isnan(float(tmp))):
			tot += tmp
		ans.append(tmp)
		cnt+=1
	for i in range(0,len(ans)):
		if (isnan(float(ans[i]))):
			ans[i] = ((tot*(1.0))/cnt)
			#ans[i] = 0
	return ans

def Register(lis, t):
	global idsList
	global le
	for val in lis:
		if (val == val):
			if (val not in le[t]):
				le[t][val] = idsList[t]
				idsList[t] += 1

def mod_clean_list(lis, t):
	ans = []
	tot = 0
	cnt = 1
	stat = 0
	if (t not in [1,4,5,7,8,9,10]):
		stat = 1
		if (le[t] is None):
			le[t] = {}
			Register(lis, t)

	for i in range(0,len(lis)):
		if (stat):
			tmp = GetCustomID(le[t], lis[i])
		else:
			tmp = float(lis[i])

		if (tmp != tmp):
			if (tmp > 0):
				tot += tmp
		ans.append(tmp)
		cnt += 1
	for i in range(0,len(ans)):
		if (isnan(float(ans[i]))):
			ans[i] = ((tot*(1.0))/cnt)
	return ans

def modify(arr):
	#return arr
	ans = []
	i=0
	while(i<len(arr)):
		if((i<=27) and (i>=23)):
			t = arr[23:28]
			#m = sorted(range(len(t)), key=t.__getitem__)
			#ans.extend(m)
			m = t.index(max(t))
			ans.append(m)
			i=27
		elif((i<=5) and (i>=1)):
			t = arr[1:6]
			#m = sorted(range(len(t)), key=t.__getitem__)
			#ans.extend(m)
			m = t.index(max(t))
			ans.append(m)
			i=5
		elif((i<=10) and (i>=6)):
			t = arr[6:11]
			#m = sorted(range(len(t)), key=t.__getitem__)
			#ans.extend(m)
			m = t.index(max(t))
			ans.append(m)
			i=10
		#elif (i in [15,16,21,28,23]):
		#	s=0
		elif(i==0):
			ans.append(arr[i])
		i+=1
	return ans

"""
1	Chromosome		Chromosome on which the mutation is present.
2	Location		Start location of the mutation in the chromosome.
3	Reference		Reference present in the genome at the location of the mutation.
4	VariantAllele		Variant allele present at the location of the mutation(reference is replaced by the variant allele).
5	Reference Length	Length of the reference.
6	VariantAllele Length	Length of the variant allele.
7	Gene			Name of the gene overlapping with the mutation.
8	DBSNP			Fraction of samples having this mutation in DBSNP.
9	1000Genome		Fraction of samples having this mutation in 1000 genome project.
10	Exome server		Fraction of samples having this mutation in exome variant server.
11	Exome consortium	Fraction of samples having this mutation in exome consortium server.
12-24	Label assigned from various sources	These columns captures label assigned from various sources to this mutation.
25	SPLICING Type		Type of missplicing caused by this variant.
26	Primates Conservation	Conservation of this mutation location in primate group.
27	Mammal Conservation	Conservation of this mutation location in mammal group.
28	Vertebrate Conservation	Conservation of this mutation location in vertebrate group.
29	Protein shortening	Protein shortening caused by this mutation.
30	isTruncating		true if this mutation is truncating for some overlapping protein.
31	isMissense		true if this mutation is missense for some overlapping protein
32	isMissenseType		true if this mutation is of missense type for some overlapping protein.
33-53	Various mutation types	these columns captures various types of mutation with respect to the various overlapping proteins.
54-60	Damaging predictions	Damaging predictions from various damaging prediction tools.
61	Damaging prediction count	Number of damaging predictions for this mutation.
62	Impacts gene expression	true if this variant can effect gene expression.	
63	functionally validated	true if this variant is functionally validated.
64	pathogenic variant 	true if there is a pathogenic variant present in the region deleted due to this variant.
65	is novel		true if this variant is a novel variant.
66	label			Label assigned to this variant.

"""

el = pandas.read_csv('train.csv', header = None)

cols = list(el.columns)
sz = len(el[cols[0]])
fl = len(cols)

cln = []
t=0
for i in cols[1:fl-1]:
	cln.append(mod_clean_list(el[i],t))
	t+=1

print idsList

print "HERE"

X = []
Y = []

blockList = [12, 13, 16, 17, 18, 21, 22, 23, 24, 33, 36, 40, 43, 44, 45, 46, 47, 50, 53, 63]
for i in range(0,len(blockList)):
	blockList[i] -= 1

print sz

for j in range(0,sz):
	tmp=[]
	for i in range(0,fl-2):
		if(i not in blockList):
			tmp.append(cln[i][j])
	X.append(tmp)
	Y.append(el[cols[66]][j])

print Y[:10]
print X[:5]
print len(X[0])

#pickle.dump([X, Y, idDict], open("data.p", "wb"))

from sklearn.feature_selection import VarianceThreshold
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

rx = []
ry = []

print len(rx)

np.set_printoptions(precision=3)

#X, Y, idDict = pickle.load(open("data.p", "rb")) 

for j in np.array(X[:3]):
	print j

#sel = VarianceThreshold(threshold=(0.95 * (1 - 0.95)))
#sel.fit_transform(X)

sz = len(X)
lis = zip(X,Y)
#shuffle(lis)
tsz = int(sz*(0.5))
tr = zip(*lis[:tsz])
ts = zip(*lis[tsz+1:])
tx, ty = tr[0], tr[1]
sx, sy = ts[0], ts[1]
print tsz, sz

# For random forests, the best candidate range is [15-37]

#for param in [10, 15, 19, 27, 33, 39, 47, 53]:
#############################

#clf = svm.SVC()
#clf = tree.DecisionTreeClassifier()
#clf = RandomForestClassifier(n_estimators=param)
#clf = AdaBoostClassifier(RandomForestClassifier(n_estimators = param), n_estimators=50)        
#clf = AdaBoostClassifier(n_estimators=73)
clf = ExtraTreesClassifier(n_estimators = 41)
#clf = NearestCentroid()
#clf = KNeighborsClassifier(n_neighbors = 19)

clf.fit(tx, ty) 

#################################

#print "TRAINING DONE for ", param
print "TRAINING DONE"

print clf.score(sx,sy)

ls = np.array(clf.feature_importances_)

for i in range(0,len(ls)):
	print i+1,ls[i],'|',

py = clf.predict(sx)
cnt = 0
tot = 0
print "Predicting DONE"
for i in range(0,len(py)):
	if(py[i]==sy[i]):
		cnt+=1
	tot+=1
print ((cnt*(1.0))/tot)
print cnt, tot

input("WAIT")

import csv

el = pandas.read_csv('test.csv', header = None)

cols = list(el.columns)
sz = len(el[cols[0]])
fl = len(cols)

#(cols[23],cols[27])=(cols[27],cols[23])					#Swapping total rallies and partial rallies

cln = []
ids = list(el[cols[0]])
t=0
for i in cols[1:]:
	cln.append(mod_clean_list(el[i],t))
	t+=1

print "HERE"

X = []
print sz

for j in range(0,sz):
	tmp=[]
	for i in range(0,fl-1):
		if(i not in blockList):
			tmp.append(cln[i][j])
	X.append(tmp)

for j in np.array(X[:3]):
	print j

py = clf.predict(X)

ans = []

for i in range(0,len(ids)):
	ans.append([ids[i],py[i]])

sorted(ans, key=lambda x: x[0])

with open('results.csv', 'wb') as testfile:
    csv_writer = csv.writer(testfile)
    csv_writer.writerow(["id", "predicted"])
    for x in ans:
        csv_writer.writerow([x[0],x[1]])
