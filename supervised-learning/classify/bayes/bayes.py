from numpy import *
from math import log

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], ['stop', 'posting', 'stupid', 'worthless', 'garbage'], ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) 
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1.0
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numWords = len(trainMatrix[0])
    numTrainDoc = len(trainMatrix)
    pAbusive = sum(trainCategory)/float(numTrainDoc)
    p0Num = ones(numWords); p1Num = ones(numWords);
    p0Denom = 2.0; p1Denom = 2.0;
    for i in range(numTrainDoc):
        if trainCategory[i] == 0:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        else:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
    p1Vec = zeros(numWords)
    p0Vec = zeros(numWords)
    for i in range(numWords):
        p1Vec[i] = log(p1Num[i]/p1Denom)
        p0Vec[i] = log(p0Num[i]/p0Denom)
    return p0Vec, p1Vec, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0: return 1
    else: return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    import re
    regEx = re.compile('\\W+')
    listOfTokens = regEx.split(bigString)
    print(listOfTokens)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
	
def spamTest():
    classList = []; docList = []; fullText = []
    for i in range(1, 26):
        fr = open('email/spam/%d.txt' % i, encoding='cp1250')
        wordStr = fr.read()
        wordList = textParse(wordStr)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        fr = open('email/ham/%d.txt' % i, encoding='cp1250')
        wordStr = fr.read()
        wordList = textParse(wordStr)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(wordList)
    trainingSet = [i for i in range(50)]; testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0.0
    for docIndex in testSet:
        wordVect = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVect), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1.0
    print("The error rate is: ", errorCount/len(testSet))
