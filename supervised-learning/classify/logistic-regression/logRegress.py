from numpy import *

def loadDataSet():
	dataMat = []; labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

def sigmoid(inX):
	return 1.0/(1.0+exp(-inX))

def gradAscent(dataMatIn, classLabels):
	dataMat = mat(dataMatIn)
	matLabels = mat(classLabels).transpose()
	_, n = shape(dataMat)
	alpha = 0.001
	maxCycles = 500
	weights = ones((n, 1))
	for i in range(maxCycles):
		errors = (matLabels - sigmoid(dataMat * weights))
		weights = weights + alpha * dataMat.transpose() * errors
	return weight

