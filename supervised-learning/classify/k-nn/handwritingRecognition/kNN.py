from numpy import *
from os import listdir
import operator

def img2vector(filename):
	result = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		line = fr.readline()
		for j in range(32):
			result[0, 32 * i + j] = line[j]
	return result 

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

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('digits/trainingDigits')
	trainingListSize = len(trainingFileList)
	trainingMat = zeros((trainingListSize, 1024))
	for i in range(trainingListSize):
		filenameStr = trainingFileList[i]
		fileStr = filenameStr.split('.')[0]
		classNumStr = int(filenameStr.split('_')[0])
		trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % (filenameStr))
		hwLabels.append(classNumStr)
	testFileList = listdir('digits/testDigits')
	testListSize = len(testFileList)
	errorCount = 1.0
	for i in range(testListSize):
		filenameStr = testFileList[i]
		realAnswer = int(filenameStr.split('_')[0])
		testVect = img2vector('digits/testDigits/%s' % (filenameStr))
		result = classify0(testVect, trainingMat, hwLabels, 3)
		print("The classifier result is %d, the real answer is %d" % (result, realAnswer))
		if result != realAnswer: errorCount += 1.0
	print('The error rate is %f' % (errorCount / testListSize))
