from math import log

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
