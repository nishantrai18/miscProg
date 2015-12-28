# Stop word Removal
# Lemmatize or stem
# Consider unigrams or bigrams (Probably no)
# Reduce to 25 dimension space (Using NMF)
# Use random forest to get the probabilities of the classes
# Might use TFIDF

import operator
import re
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree

np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)

def normalize(v):
	tmpSum = np.sum(v**2)
	if (tmpSum == 0):
		return v
	return v/tmpSum

vocab = {}
numTrain = 0
numQuery = 0
docTerm = []
docs = []
labels = []
freqLabel = {}
testDocs = []

cnt = 0
with open('labeler_sample.in','r') as f:
	for line in f:
		if (cnt == 0):
			numTrain, numQuery = int(line.split()[0]), int(line.split()[1])
			cnt+=1
			continue
		if (cnt<=(2*numTrain)):
			#print re.split('\W+', line.lower())
			if (cnt%2==0):
				words = []
				for word in re.split('\W+', line.lower()):
					if (len(word)>1):
						if word.lower() in vocab:
							vocab[word.lower()] += 1
						else:
							vocab[word.lower()] = 1
						words.append(word.lower())
				docs.append(words)
				#print re.split('\W+', line.lower())
				#print "DOC VEC ", words
			else:
				#print line.split()
				tmpLabel = [int(x) for x in line.split() if (len(x)>0)]
				labels.append(tmpLabel)
				#print tmpLabel
		else:
			words = []
			for word in re.split('\W+', line.lower()):
				if (len(word)>1):
					words.append(word.lower())
			testDocs.append(words)
		cnt+=1

freqList = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)

#print len(vocab)

wordList = {}
ids = 0
for i in range(100,2000):
	wordList[freqList[i][0]] = ids
	ids+=1

for i in range(0,len(docs)):
	for j in labels[i]:
		if j in freqLabel:
			freqLabel[j]+=1
		else:
			freqLabel[j]=1
freqLabelList = sorted(freqLabel.items(), key=operator.itemgetter(1), reverse=True)

numFeat = len(wordList)
#print wordList
#print docs

for i in range(0, len(docs)):
	term = np.zeros(numFeat)
	for j in docs[i]:
		if (j in wordList):
			term[wordList[j]]+=1
	docTerm.append(term)
docTerm = np.array(docTerm)

#print "HERE"

numDim = 250
#xn = PCA(n_components=25).fit_transform(docTerm)
reducer = TruncatedSVD(n_components=numDim).fit(docTerm)
#reducer = NMF(n_components=numDim, max_iter=100).fit(docTerm)
lowSp = reducer.transform(docTerm)

X = []
Y = []
for i in range(0,len(docTerm)):
	for j in labels[i]:
		X.append(lowSp[i])
		Y.append(j)
X = np.array(X)
Y = np.array(Y)

for r in lowSp:
	r = normalize(r)

tree = KDTree(lowSp);
#print len(X),len(Y)

#clf = RandomForestClassifier(n_estimators=19)
#clf = clf.fit(X, Y)

#print "HERE"

tdocTerm = []

for i in range(0, len(testDocs)):
	term = np.zeros(numFeat)
	for j in docs[i]:
		if (j in wordList):
			term[wordList[j]]+=1
	tdocTerm.append(term)
tdocTerm = np.array(tdocTerm)

lowTsp = reducer.transform(tdocTerm)

for r in lowTsp:
	r = normalize(r)

numNeigh = 20

for i in range(0,len(lowTsp)):
	probs = {}
	for m in range(0,251):
		probs[m]=0
	dist, ind = tree.query(lowTsp[i], k=numNeigh)
	dist = dist[0]
	ind = ind[0]
	maxDist = max(dist)
	for t in range(0,len(ind)):
		num = len(labels[ind[t]])
		for x in labels[ind[t]]:
			probs[x] += 1.0	#*(maxDist-dist[t])
	ids = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)
	ans = []
	for t in ids[:10]:
		if (probs[t] == 0):
			break
		else:
			ans.append(t)
	if (len(ans) < 10):
		j = 0
		while (len(ans) < 10):
			if (freqLabelList[j][0] not in ans):
				ans.append(freqLabelList[j][0])
			j+=1
	for m in ans:
		print m,
	print


"""
probs = clf.predict_proba(lowTsp)

for i in range(0,len(lowTsp)):
	ids = sorted(range(len(probs[i])), key=lambda k: probs[i][k], reverse=True)
	ans = []
	for t in ids[:10]:
		if (probs[i][t] == 0):
			break
		else:
			ans.append(t+1)
	if (len(ans) < 10):
		j = 0
		while (len(ans) < 10):
			if (freqLabelList[j][0] not in ans):
				ans.append(freqLabelList[j][0])
			j+=1
	for m in ans:
		print m,
	print
"""

"""
for j in probs[:10]:
	suma = 0
	for t in j:
		if t>0 :
			print "%.2f" % t,
	print
"""