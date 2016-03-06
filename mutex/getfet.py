import pandas
from math import isnan
from random import shuffle
import numpy as np

el = pandas.read_csv('train.csv', header = None)

cols = list(el.columns)
sz = len(el[cols[0]])
fl = len(cols)

# (cols[24],cols[28])=(cols[28],cols[24])					#Swapping total rallies and partial rallies

def categs(arr):
	if('CENTAUR' in arr):
		return 0
	elif('COSMOS' in arr):
		return 1
	elif('ODYSSEY' in arr):
		return 2
	elif('TOKUGAWA' in arr):
		return 3
	else:
		return 4

def invcateg(num):
	if(num==2):
		return 'ODYSSEY'
	elif(num==4):
		return 'EBONY'
	elif(num==0):
		return 'CENTAUR'
	elif(num==1):
		return 'COSMOS'
	else:
		return 'TOKUGAWA'

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

def clean(arr,t):
	#print arr
	if(t==11):
		if('Service' in arr):
			return 14
		elif ('Doctor' in arr):
			return 19
 		elif('Social Worker' in arr):
			return 8
		elif('House Husband' in arr):
			return 4
		elif('Factory Mager' in arr):
			return 16
		elif('Middle Magement' in arr):
			return 15
		elif('Nurse' in arr):
			return 5
		elif('Self Employed' in arr):
			return 13
		elif('Director' in arr):
			return 23
		elif('Dentist' in arr):
			return 17
		elif('Student' in arr):
			return 2
		elif('Others' in arr):
			return 12
		elif('Senior Magement' in arr):
			return 22
		elif('Magician' in arr):
			return 1
		elif('Amabassador' in arr):
			return 20
		elif('Pilot' in arr):
			return 17
		elif('Politician' in arr):
			return 21
		elif('Consultant' in arr):
			return 18
		elif('Military' in arr):
			return 9
		elif('Janitor' in arr):
			return 3
		elif('Teacher' in arr):
			return 6
		elif('Chef' in arr):
			return 7
		elif('Scientist' in arr):
			return 11
		else:
			return 10
	elif (t==14):
		ls = arr.split('-')
		if (len(ls)==2):
			return ((float(ls[0])+float(ls[1]))/2)
		else:
			ls=ls[0].split('+')
			return float(ls[0])
	elif(t==0):
		return categs(arr)
	elif (t==21):
		if ('primary' in arr):
			return 0
		elif ('MBA' in arr):
			return 1
		else:
			return 2
	elif ((t<6) and (t>0)):
		ls = arr.split('$')
		return float(ls[1].replace(',',''))
	elif ((t<28) and (t>20)):
		#print arr, float(arr)
		if(isnan(float(arr))):
			return 0
		else:
			return float(arr)
	elif (isinstance(arr, float) or isinstance(arr, int) ):
		return float(arr)
	else:
		print "EXCPT"
		return 0
	#Some conditions

def clean_list(lis, t):
	ans = []
	tot = 0
	cnt = 0
	for i in lis:
		tmp = clean(i,t)
		if (not isnan(float(tmp))):
			tot += tmp
		ans.append(tmp)
		cnt+=1
	for i in range(0,len(ans)):
		if (isnan(float(ans[i]))):
			#ans[i] = ((tot*(1.0))/cnt)
			ans[i] = 0
	return ans

def new_clean_list(lis, t):
	ans = []
	tot = 0
	cnt = 1 			# To avoid division errors
	stat = 0

	for i in lis:
		ans.append(i)
		if (not isinstance(i, (int, long, float))):
			stat = 1
			continue
		if (not isnan(float(i))):
			tot += i
			cnt += 1

	for i in range(0,len(ans)):
		if (ans[i] != ans[i]):
			if (stat):
				ans[i] = 'NULL'	
			else:
				ans[i] = ((tot*(1.0))/cnt)
	return ans

cln = []
t=0
for i in cols[1:fl-1]:
	cln.append(new_clean_list(el[i],t))
	t+=1

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

print "HERE"

X = []
Y = []

print sz

for j in range(0,sz):
	tmp=[]
	for i in range(0,fl-2):
		if(i not in []):
			tmp.append(cln[i][j])
	X.append(tmp)
	Y.append(el[cols[66]][j])

print Y[:10]
print X[:5]
print len(X[0])

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


"""
for j in range(0,len(X)):
	if(X[j][29]>0):
		rx.append(X[j][:-1])
		ry.append(X[j][29])
"""

print len(rx)

#reg  = linear_model.LinearRegression()

#reg.fit(rx, ry)

print "TRAINING regressor done"

#for j in range(0,len(X)):
#	if(X[j][29]==0):
#		X[j][29] = reg.predict([X[j][:-1]])

print "SOME done"

np.set_printoptions(precision=3)

for j in np.array(X[:3]):
	print j

#sel = VarianceThreshold(threshold=(0.95 * (1 - 0.95)))
#sel.fit_transform(X)

lis = zip(X,Y)
shuffle(lis)
tsz = int(sz*(0.79))
tr = zip(*lis[:tsz])
ts = zip(*lis[tsz+1:])
tx, ty = tr[0], tr[1]
sx, sy = ts[0], ts[1]
print tsz, sz


#############################

#clf = svm.SVC()
#clf = tree.DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators=33)
#clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=19), n_estimators=50)        
#clf = AdaBoostClassifier(n_estimators=73)
#clf = ExtraTreesClassifier(n_estimators=33)
#clf = NearestCentroid()
#clf = KNeighborsClassifier(n_neighbors = 19)

clf.fit(tx, ty) 

#################################


print "TRAINING DONE"

print clf.score(sx,sy)

ls = np.array(clf.feature_importances_)

for i in range(0,len(ls)):
	print i+1,ls[i]

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


'''

import csv

el = pandas.read_csv('Leaderboard_Dataset.csv')
#el = pandas.read_csv('Training_Dataset.csv')


cols = list(el.columns)
sz = len(el[cols[0]])
fl = len(cols)

(cols[23],cols[27])=(cols[27],cols[23])					#Swapping total rallies and partial rallies

cln = []
ids = list(el[cols[0]])
t=0
for i in cols[1:]:
	cln.append(clean_list(el[i],t))
	t+=1

for i in range(1,len(cols)):
	if((i<=28) and (i>=24)):
		for j in range(0,len(cln[i-1])):
			if(cln[22][j]>0):
				cln[i-1][j] = ((cln[i-1][j]*(1.0))/(cln[22][j]))
			else:
				cln[i-1][j] = 0
	elif((i<=6) and (i>=2)):
		for j in range(0,len(cln[i-1])):
			tot = 0
			for m in range(1,6):
				tot += cln[m][j]
			for m in range(2,7):
				if(tot>0):
					cln[m-1][j] = ((cln[m-1][j]*(1.0))/tot)
				else:
					cln[m-1][j] = 0
			i=6
	elif((i<=11) and (i>=7)):
		for j in range(0,len(cln[i-1])):
			tot = 0
			for m in range(6,11):
				tot += cln[m][j]
			for m in range(7,12):
				if(tot>0):
					cln[m-1][j] = ((cln[m-1][j]*(1.0))/tot)
				else:
					cln[m-1][j] = 0
			i=11

print "HERE"

X = []
print sz

for j in range(0,sz):
	tmp=[]
	for i in range(0,fl-1):
		if(i not in []):
			tmp.append(cln[i][j])
	X.append(modify(tmp))

for j in range(0,len(X)):
	if(X[j][17]==0):
		X[j][17] = reg.predict([X[j][:-1]])

print "SOME done"

np.set_printoptions(precision=3)

for j in np.array(X[:3]):
	print j

py = clf.predict(X)

ans = []

for i in range(0,len(ids)):
	ans.append([ids[i],py[i]])

sorted(ans, key=lambda x: x[0])

with open('Prometheus_IITKanpur_1.csv', 'wb') as testfile:
    csv_writer = csv.writer(testfile)
    for x in ans:
        csv_writer.writerow([x[0],invcateg(x[1])])


"""
df = pandas.DataFrame({'A' : [x[0] for x in ans],'B' : [invcateg(x[1]) for x in ans]})

df.to_csv('result.csv')

print df['A'][:100]
print df['B'][:100]
"""
'''
