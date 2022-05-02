from numpy import *
import operator

def file2matrix(filename):
	fr = open(filename)
	linesLen = len(fr.readlines())
	returnMat = zeros((linesLen, 3))
	classLabelVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()	
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector

def autoNorm(dataSet):
	minValues = dataSet.min(0)
	maxValues = dataSet.max(0)
	rangeValues = maxValues - minValues
	normData = zeros(shape(dataSet))
	rowSize = dataSet.shape[0]
	normData = dataSet - tile(minValues, (rowSize, 1))
	normData = normData / tile(rangeValues, (rowSize, 1))
	return normData, rangeValues, minValues

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	sortedDistIndex = distances.argsort()
	classCount = {}
	for i in range(k):
		voteILabel = labels[sortedDistIndex[0]]
		classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

	return sortedClassCount[0][0]

def testDatingClass():
	hoRatio = 0.1
	datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
	normData, rangeValues, minValues = autoNorm(datingDataMat)
	setTestSize = normData.shape[0] 
	testRowsSize = int(hoRatio * setTestSize)
	errorCount = 0.0
	for i in range(testRowsSize):
		classifyResult = classify0(normData[i,:], normData[testRowsSize:setTestSize, :], datingLabels[testRowsSize:setTestSize], 3)
		print("the classifier came back with: %d, the real answer is: %d" % (classifyResult, datingLabels[i]))
		if (classifyResult != datingLabels[i]): errorCount += 1
	print("The total error rate is: %f" % (errorCount/float(testRowsSize)))  

def classifyPerson():
	resultList = ['not at all', 'in small dose', 'in large dose']
	gameSpentPercent = float(input("Game: "))
	ffMiles = float(input("Flyer miles: "))
	iceCream = float(input("Ice Cream:" ))
	datingMatData, datingLabels = file2matrix("datingTestSet2.txt")
	normData, rangeValues, minValues = autoNorm(datingMatData)
	personArray = array([ffMiles, gameSpentPercent, iceCream])
	classifyResult = classify0(personArray, normData, datingLabels, 3)
	print("%s" % (resultList[classifyResult - 1]))
