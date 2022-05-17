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
	return weights

def stocGradAscent0(dataMatIn, classLabels):
	m, n = shape(dataMatIn)
	weights = ones(n)
	alpha = 0.001
	for i in range(m):
		errors = (classLabels[i] - sigmoid(sum(dataMatIn[i] * weights))) 
		weights = weights + alpha * dataMatIn[i] * errors
	return weights

def stocGradAscent1(dataMatIn, classLabels, iterNumber=150):
	m, n = shape(dataMatIn)
	weights = ones(n)
	for i in range(iterNumber):
		dataIndex = [t for t in range(m)]
		for j in range(m):
			alpha = 4.0/(1.0 + i + j) + 0.01
			randIndex = int(random.uniform(0, len(dataIndex)))
			errors = (classLabels[randIndex] - sigmoid(sum(dataMatIn[randIndex] * weights))) 
			weights = weights + alpha * dataMatIn[randIndex] * errors
			del(dataIndex[randIndex]) 
	return weights

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    x1Cord = []; y1Cord = []
    x2Cord = []; y2Cord = []
    for i in range(n):
        if (labelMat[i] == 1):
            x1Cord.append(dataArr[i, 1]); y1Cord.append(dataArr[i, 2])
        else:
            x2Cord.append(dataArr[i, 1]); y2Cord.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1Cord, y1Cord, s=30, color='red', marker='s')
    ax.scatter(x2Cord, y2Cord, s=30, color='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - x * weights[1]) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
