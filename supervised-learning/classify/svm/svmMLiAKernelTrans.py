from numpy import *

def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def kernelTrans(X, A, kTup):
    m, _ = shape(X)
    K = mat(zeros((m, 1)))
    if (kTup[0] == 'lin'): K = X * A.T
    elif (kTup[0] == 'rbf'):
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K/-1*kTup[1]**2)
    else: raise NameError('Sang We have a problem ---- That kernel is not recognized')
    return K

class optStruct():
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.m, _ = shape(self.X)
        self.C = C
        self.tol = toler
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup) 

def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k]) + oS.b
    Ek = fXk - oS.labelMat[k]
    return Ek

def selectJrand(i, m):
    j = i
    while (i == j):
        j = float(random.uniform(0, m))
    return j

def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if len(validEcacheList) > 0:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ej)
            if (deltaE > maxDeltaE): maxDeltaE = deltaE; maxK = k; Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, k)
        return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def clipAlpha(alpha, H, L):
    if alpha > H: alpha = H
    if alpha < L: alpha = L
    return alpha

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((Ei * oS.labelMat[i] < -oS.tol) and (oS.alphas[i] < oS.C)) or ((Ei * oS.labelMat[i] > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i])
            H = min(oS.C, oS.alphas[j] + oS.alphas[i] - oS.C)
        if L == H: print("L == H"); return 0
        eta = 2 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0: print("eta >= 0"); return 0
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        oS.alphas[j] = oS.alphas[j] - oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j is not moving enough"); return 0
        oS.alphas[i] = oS.alphas[j] + oS.labelMat[i] * oS.labelMat[j] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (alphaIold - oS.alphas[i]) * oS.K[i, i] - oS.labelMat[j] * (alphaJold - oS.alphas[j]) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[j] * (alphaJold - oS.alphas[j]) * oS.K[j, j] - oS.labelMat[i] * (alphaIold - oS.alphas[i]) * oS.K[i, j]
        if (oS.alphas[i] > 0 and oS.alphas[i] < oS.C): b = b1
        elif (oS.alphas[j] > 0 and oS.alphas[j] < oS.C): b = b2
        else: b = (b1 + b2) / 2.0
        return 1.0
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('rbf', 1.3)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iterTemp = 0
    alphaPairsChanged = 0
    entireSet = True
    while (iterTemp < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if (entireSet):
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullset, iter: %d, i: %d, alphaPairsChanged: %d" % (iterTemp, i, alphaPairsChanged))
            iterTemp += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d, i: %d, alphaPairsChanged: %d" % (iterTemp, i, alphaPairsChanged))
            iterTemp += 1                
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iterTemp)
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    _, n = shape(X)
    w = zeros(zeros(n, 1))
    for i in range(n):
        w += multiply(alphas[i] * labelMat[i], X[i,:].T)
    return w

def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svIdx = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svIdx]
    labelSV = labelMat[svIdx]
    print("There are %d support vectors" % shape(sVs)[0])
    m, _ = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svIdx]) + b
        if (sign(predict) != sign(labelArr[i])): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    errorCount = 0
    m, _ = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svIdx]) + b
        if (sign(predict) != sign(labelArr[i])): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))
