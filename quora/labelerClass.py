import math
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

wordNetLem = WordNetLemmatizer()

def increaseDictKey (argDict, key, incNum):
    """
    Increments the value of a[key] by inc 
    Assumes deafult value of dictionary to be 0
    argDict: dictionary to work on
    key: key to update or create
    incNum: value to be increment with
    return value: None
    """
    if key in argDict:
        argDict[key] += incNum
    else:
        argDict[key] = incNum

class Query:
    def __init__(self, queryString=None):
        self.question = queryString
        self.categories = []
        self.words = []
        self.wordWeight = {}
        self.docFreq = {}

class Labeler:
    def __init__(self):
        self.dictionary = {}                #dictionary with word frequencies
        self.wordMap = {}                   #The mappings of the dictionary words to 
        self.questions = []
        self.stopwords = []                 #Set of stopwords

    def AddQuestion(self, question, categories):
        """
        Adds a new Question to the labeler's questions list
        categories: It is a string consisting of space separated IDs
        question: The question string
        return value: None
        """

        q = Query()
        q.question = question
        q.categories = categories.split(" ")
        wordList = word_tokenize(question)                    # Avoid the last character (If it's a question mark)
        goodWords = []
        for w in wordList:
            if w.lower() not in self.stopwords:
                w = self.wordNetLem.lemmatize(w, NOUN)
                goodWords.append(w)
        q.words = goodWords

        for i, w in enumerate(goodWords):
            increaseDictKey(q.docFreq, w, 1)                       #Term document frequency for the word
            if w in self.dictionary:
                self.dictionary[w] += 1
            else:
                self.dictionary[w] = 1
                self.wordMap = len(self.dictionary)
        self.queries.append(q)


def main():
    numTrain, numTest = [int(i) for i in raw_input().split(" ")]  # Training questions, Test questions
    label = Labeler()
    for i in range(0, numTrain):
        questionString = raw_input()                                        # count category_id category_id category_id...
        categoryString = raw_input()
        label.AddQuestion(questionString, categoryString)

if __name__ == '__main__':
    main()