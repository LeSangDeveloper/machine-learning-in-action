from numpy import *

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([lineArr[0], lineArr[1]])
        labelMat.append(lineArr[2])
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while (i == j):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).tranpose()
    m, n = shape(dataMatrix)
    alphas = zeros(m, 1); iterTemp = 0; b = 0
    while (iterTemp < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # f(xi) = w.xi + b
            fXi = (multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i:].T)) + b
            # E(i) = f(xi) - y(i)
            Ei = fXi - float(labelMat[i])
            if ((Ei * labelMat[i] < -toler) and (alphas[i] < C) or (Ei * labelMat[i] > toler and alphas[i] > 0)):
                j = selectJrand(i, m)

        if (alphasPairsChanged == 0): iterTemp += 1
        else: iterTemp = 0
    return b, alphas
