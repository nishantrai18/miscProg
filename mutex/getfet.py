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
numList = [1,4,5,7,8,9,10,28,60]
seqList = [2,3]
costs = {}

idx = 0
for a in ['A','T','G','C','e']:
	for b in ['A','T','G','C','e']:
		costs[a+b] = idx
		idx += 1
seqDim = idx

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
		return len(dic)
	if (key not in dic):
		return len(dic)
	return dic[key]

def clean_list(lis, t):
	ans = []
	tot = 0
	cnt = 1
	for i in range(0,len(lis)):
		tmp = GetID(lis[i])
		if (not isnan(float(tmp))):
			tot += tmp
			cnt+=1
		ans.append(tmp)
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
	if (t not in numList):
		stat = 1
		if (le[t] is None):
			le[t] = {}
			Register(lis, t)

	for i in range(0,len(lis)):
		if (stat):
			tmp = GetCustomID(le[t], lis[i])
		else:
			tmp = float(lis[i])

		if (tmp == tmp):
			tot += tmp
			cnt += 1
		ans.append(tmp)

	for i in range(0,len(ans)):
		if (isnan(float(ans[i]))):
			ans[i] = ((tot*(1.0))/cnt)
	return ans

def GetOneHot(val, dim, binStat = 0):
	ans = [0]*dim
	if ((val != val) or (val < 0)):
		ans[-1] = 1
	elif (binStat):
		ans = bin(val)[2:].zfill(dim)
		ans = list(ans)
		ans = [int(x) for x in ans]
	else:
		ans[val] = 1
	return ans

def GetLen(dic):
	if (not dic):
		return 0
	else:
		return len(dic)

def DecodeGram(gram):
	if (len(gram) == 1):
		return 

def GetAvgList(lis):
	cnt = 1
	tot = 0
	for i in range(0,len(lis)):
		tmp = float(lis[i])
		if (tmp == tmp):
			tot += tmp
			cnt += 1
	return (tot/cnt)

def GetSeqFet(seq):
	fet = [0]*seqDim
	for i in range(0,len(seq)):
		if (i==0):
			fet[costs['e'+seq[i]]] += 1
		else:
			fet[costs[seq[i-1] + seq[i]]] += 1
		if (i == (len(seq)-1)):
			fet[costs[seq[i]+'e']] += 1
	tmpSum = sum(fet)*(1.0)
	fet = [x/tmpSum for x in fet]
	return fet

def mult_clean_list(lis, t):
	ans = []
	tot = 0
	cnt = 1
	stat = 0
	if (t not in numList):
		stat = 1
		if (le[t] is None):
			le[t] = {}
			Register(lis, t)
	dim = GetLen(le[t]) + 1				# To account for NaNs

	if (dim > 30):
		return [mod_clean_list(lis, t)], [t]

	if (False):
		dim = int(np.log2(dim) + 1)

	if (t in seqList):
		dim = seqDim

	if (dim == 1):
		dim += 1

	# if (not stat):
	# 	tmpAvg = GetAvgList(lis)

	for i in range(0,len(lis)):
		if (stat):
			if (t in seqList):
				oneHot = GetSeqFet(lis[i])
			else:
				tmp = GetCustomID(le[t], lis[i])
				oneHot = GetOneHot(tmp, dim, 0)
		else:
			tmp = float(lis[i])
			# if (tmp == tmp):
			# 	oneHot = [tmp]
			# else:
			# 	oneHot = [tmpAvg]
			if (tmp != tmp):
				oneHot = [0, 0]
			else:
				oneHot = [tmp, 1]
		ans.append(oneHot)
	return map(list, zip(*ans)), [t]*dim

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

parList = []
cln = []
t=0
for i in cols[1:fl-1]:
	tmpList, par = mult_clean_list(el[i],t)
	# tmpList = mod_clean_list(el[i],t)
	cln.extend(tmpList)				# The function should return a list of lists
	parList.extend(par)
	# parList.extend([t])
	t += 1

# for x in cln:
# 	print x[:10]

print idsList

print "HERE"


X = []
Xt = [[], [], []]
Y = []

# blockList = [12, 13, 16, 17, 18, 21, 22, 23, 24, 33, 36, 40, 43, 44, 45, 46, 47, 50, 53, 63]
blockList = []

goodList1 = range(11,24)
goodList2 = range(32,53)
goodList3 = range(53,60)


for i in range(0,len(blockList)):
	blockList[i] -= 1

print sz

for j in range(0,sz):
	tmp = []
	tmp1 = []
	tmp2 = []
	tmp3 = []
	for i in range(0,len(cln)):
		if (parList[i] in goodList1):
			tmp1.append(cln[i][j])			
		elif (parList[i] in goodList2):
			tmp2.append(cln[i][j])			
		elif (parList[i] in goodList3):
			tmp3.append(cln[i][j])
		elif (parList[i] not in blockList):
			tmp.append(cln[i][j])			
	X.append(tmp)
	Xt[0].append(tmp1)
	Xt[1].append(tmp2)
	Xt[2].append(tmp3)	
	Y.append(el[cols[66]][j])

print Y[:10]
print X[:3]
print len(X[0])

#pickle.dump([X, Y, idDict], open("data.p", "wb"))

'''
Possible improvements:
Consider different chunks of same value as different vectors i.e. 12-24, 33-54, 54-60
Consdier one hot encodings for a few attributes with small ranges	(DONE)
Consider 2-grams and less for 3 and 4 	(DONE)
Ignore effect of 5 and 6 (Or refine its use)
Variance Thresholding	(DONE)
'''

from sklearn.feature_selection import VarianceThreshold
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

ratio = 0.5

clfs = []
pred = []

for t in range(0,3):
	sz = len(Xt[t])
	lis = zip(Xt[t],Y)
	tsz = int(sz*(ratio))
	tr = zip(*lis[:tsz])
	ts = zip(*lis[tsz+1:])
	tx, ty = np.array(tr[0]), np.array(tr[1])
	sx, sy = np.array(ts[0]), np.array(ts[1])
	print tsz, sz
	clfs.append(RandomForestClassifier(n_estimators = 70))
	clfs[t].fit(tx,ty)
	pred.append(clfs[t].predict(Xt[t]))

for i in range(0,len(X)):
	for t in range(0,3):
		X[i].extend(GetOneHot(pred[t][i],10,0))

print X[:3]

rx = []
ry = []

print len(rx)

np.set_printoptions(precision=3)

#X, Y, idDict = pickle.load(open("data.p", "rb")) 

# for j in np.array(X[:3]):
# 	print j

sel = VarianceThreshold(threshold=(0.999 * (1 - 0.999)))
X = sel.fit_transform(X)

print len(X[0])

sz = len(X)
lis = zip(X,Y)
#shuffle(lis)
tsz = int(sz*ratio)
tr = zip(*lis[:tsz])
ts = zip(*lis[tsz+1:])
tx, ty = np.array(tr[0]), np.array(tr[1])
sx, sy = np.array(ts[0]), np.array(ts[1])
print tsz, sz

print tx
print ty[:100]

# For random forests, the best candidate range is [15-37]

#for param in [10, 15, 19, 27, 33, 39, 47, 53]:
#############################

# clf = svm.LinearSVC()
# clf = AdaBoostClassifier(n_estimators=13)
# clf = NearestCentroid()
# clf = KNeighborsClassifier(n_neighbors = 19, n_jobs = 4)
# clf = RandomForestClassifier(n_estimators = 41, class_weight = 'balanced')

clf = RandomForestClassifier(n_estimators = 41)
# clf = AdaBoostClassifier(RandomForestClassifier(n_estimators = 43), n_estimators = 65)        

# clf1 = tree.DecisionTreeClassifier()
# clf2 = RandomForestClassifier(n_estimators = 541)
# clf4 = AdaBoostClassifier(RandomForestClassifier(n_estimators = 43), n_estimators = 65)        
# clf3 = ExtraTreesClassifier(n_estimators = 441)

# clf = VotingClassifier(estimators=[('c1', clf1), ('c2', clf2), ('c3', clf3), ('c4', clf4)], voting='soft', weights=[1,1,1,1])

#################################

clf.fit(tx,ty)

#print "TRAINING DONE for ", param
print "TRAINING DONE for "

print clf.score(sx,sy)

# ls = np.array(clf.feature_importances_)

# for i in range(0,len(ls)):
# 	print i+1,ls[i]

# py = clf.predict(sx)
# cnt = 0
# tot = 0
# print "Predicting DONE"
# for i in range(0,len(py)):
# 	if(py[i]==sy[i]):
# 		cnt+=1
# 	tot+=1
# print ((cnt*(1.0))/tot)
# print cnt, tot

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
for i in cols[1:fl]:
	tmpList, par = mult_clean_list(el[i],t)
	cln.extend(tmpList)
	t += 1
print "HERE"

X = []
print sz

for j in range(0,sz):
	tmp=[]
	for i in range(0,len(cln)):
		if(parList[i] not in blockList):
			tmp.append(cln[i][j])
	X.append(tmp)

X = sel.transform(X)

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
