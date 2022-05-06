from math import log
import operator

def createDataSet():
	dataSet = [[1, 1, 'yes'],
		[1, 1, 'yes'],
		[1, 0, 'no'],
		[0, 1, 'no'],
		[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVect in dataSet:
		if featVect[axis] == value:
			reducedFeatVect = featVect[:axis]
			reducedFeatVect.extend(featVect[axis+1:])
			retDataSet.append(reducedFeatVect)
	return retDataSet

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	countLabels = {}
	for featVect in dataSet:
		currentLabel = featVect[-1]
		if currentLabel not in countLabels.keys():
			countLabels[currentLabel] = 0.0
		countLabels[currentLabel] += 1.0
	entrophyResult = 0.0
	for _, value in countLabels.items():
		prob = float(value)/numEntries
		entrophyResult -= prob * log(prob, 2)
	return entrophyResult

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntrophy = calcShannonEnt(dataSet)
	bestFeature = -1; bestInfoGain = 0.0
	for i in range(numFeatures):
		listFeatures = [example[i] for example in dataSet]
		uniqueVals = set(listFeatures)
		newEntrophy = 0.0
		for val in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, val)
			prob = float(len(subDataSet))/len(dataSet)
			newEntrophy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntrophy - newEntrophy
		if bestInfoGain < infoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0.0
        classCount += 1.0
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    bestFeatValues = [example[bestFeat] for example in dataSet]
    uniqueBestFeatValues = set(bestFeatValues)
    for value in uniqueBestFeatValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
