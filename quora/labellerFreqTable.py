# Stop word Removal
# Lemmatize or stem
# Consider unigrams or bigrams (Probably no)
# Reduce to 25 dimension space (Using NMF)
# Use random forest to get the probabilities of the classes
# Might use TFIDF

import operator
import re
import numpy as np
from random import shuffle
from math import log
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer

#wordnet_lem = WordNetLemmatizer()

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)

vocab = {}					#Initial vocabulary with word frequencies
idf = {}					#Contains the idf of the words with each sentence considered as a document
varWeight = {}				#Contains the variance of the probability
numTrain = 0
numQuery = 0
numLabels = 250
docTerm = []
docs = []					#Contains the sentences with the involved words in them
labels = []					#Contains the associated labels with each sentence/
freqLabel = {}				#Conatins the frequency of the labels
testDocs = []				#Contains the test sentences i.e. the ones for which we need to find the labels

cnt = 0

#FIle version for local tests
"""
with open('labeler_sample.in','r') as f:
	for line in f:
		if (cnt == 0):
			numTrain, numQuery = int(line.split()[0]), int(line.split()[1])
			cnt+=1
			continue
		if (cnt<=(2*numTrain)):
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
			else:
				tmpLabel = [int(x) for x in line.split() if (len(x)>0)]
				labels.append(tmpLabel)
		else:
			words = []
			for word in re.split('\W+', line.lower()):
				if (len(word)>1):
					words.append(word.lower())
			testDocs.append(words)
		cnt+=1
"""

#For reading files from input buffer
line = raw_input()
numTrain, numQuery = int(line.split()[0]), int(line.split()[1])

for i in range(0,numTrain):
	line = raw_input()
	tmpLabel = [int(x) for x in line.split()[1:] if (len(x)>0)]
	labels.append(tmpLabel)
	line = raw_input()
	line = line.lower()
	words = []
	#words = word_tokenize(line)  
	#words = [wordnet_lem.lemmatize(word) for word in words]
	for word in re.split('\W+', line.lower()):
		#word = wordnet_lem.lemmatize(word)
		if (len(word)>1):
			if word.lower() in vocab:
				vocab[word.lower()] += 1
			else:
				vocab[word.lower()] = 1
			words.append(word.lower())
	for word in words:
		if word in vocab:
			vocab[word] += 1
		else:
			vocab[word] = 1
	docs.append(words)

for i in range(0,numQuery):
	line = raw_input()
	line = line.lower()
	words = []
	#words = word_tokenize(line)
	#words = [wordnet_lem.lemmatize(word) for word in words]
	for word in re.split('\W+', line.lower()):
		#word = wordnet_lem.lemmatize(word)
		if (len(word)>1):
			words.append(word.lower())
	testDocs.append(words)

#Get the idf's for each word

for i in docs:
	tmpDict = {}
	for j in i:
		tmpDict[j] = 1
	for j in tmpDict.keys():
		if j in idf:
			idf[j] += 1
		else:
			idf[j] = 1

for x in idf.keys():
	idf[x] = log(numTrain*(1.0)/(idf[x]+1))

#print idf

#freqList = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)				#Sort the vocabulary to get the most frequent worlds
freqList = sorted(idf.items(), key=operator.itemgetter(1), reverse=True)				#Sort the vocabulary to get the most frequent worlds

wordList = {}															#Contains the list of top words which would be considered as our dictionary 
																		#Should use tfidf for pruning instead
ids = 0
for i in range(0,len(freqList)):
	if freqList[i][0] in stopwords:
		continue
	wordList[freqList[i][0]] = ids
	ids+=1

for i in range(0,len(docs)):
	for j in labels[i]:
		if j in freqLabel:
			freqLabel[j]+=1
		else:
			freqLabel[j]=1
freqLabelList = sorted(freqLabel.items(), key=operator.itemgetter(1), reverse=True)			#Get the list of top labels

numFeat = len(wordList)						#Number of features involved

wordLabel = {}						#Coocurrence matrix (probabilities) for the dictionary words and labels

for i in range(0, len(docs)):					#Create the matrix
	for j in docs[i]:
		if j in wordList:
			if j not in wordLabel:
				wordLabel[j] = np.zeros(numLabels+1)
			for l in labels[i]:
				wordLabel[j][l] += 1

for j in wordLabel:										#Normalize to get probabilities
	tmpSum = np.sum(wordLabel[j])
	wordLabel[j] = wordLabel[j]/tmpSum
	varWeight[j] = wordLabel[j].var()**0.6

docTerm = np.array(docTerm)							#Equivalent to our data matrix X

for i in range(0,len(testDocs)):					#Predict the labels of the test set
	probs = np.zeros(numLabels+1)
	for j in testDocs[i]:
		if j in wordList:
			probs = probs + wordLabel[j]*(1.0)*varWeight[j]
	#print probs
	ids = (probs.argsort())[::-1]
	#print ids
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