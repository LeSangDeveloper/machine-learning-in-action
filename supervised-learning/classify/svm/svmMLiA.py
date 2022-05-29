from numpy import *

class optStruct():
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.m, _ = shape(self.X)
        self.C = C
        self.tol = toler
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))

def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
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
        eta = 2 * (oS.X[i,:] * oS.X[j,:].T) - (oS.X[i,:] * oS.X[i,:].T) - (oS.X[j,:]*oS.X[j,:].T)
        if eta >= 0: print("eta >= 0"); return 0
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        oS.alphas[j] = oS.alphas[j] - oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j is not moving enough"); return 0
        oS.alphas[i] = oS.alphas[j] + oS.labelMat[i] * oS.labelMat[j] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (alphaIold - oS.alphas[i]) * (oS.X[i,:] * oS.X[i,:].T) - oS.labelMat[j] * (alphaJold - oS.alphas[j]) * (oS.X[i,:] * oS.X[j,:].T)
        b2 = oS.b - Ej - oS.labelMat[j] * (alphaJold - oS.alphas[j]) * (oS.X[j,:] * oS.X[j,:].T) - oS.labelMat[i] * (alphaIold - oS.alphas[i]) * (oS.X[i,:] * oS.X[j,:].T)
        if (oS.alphas[i] > 0 and oS.alphas[i] < oS.C): b = b1
        elif (oS.alphas[j] > 0 and oS.alphas[j] < oS.C): b = b2
        else: b = (b1 + b2) / 2.0
        return 1.0
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
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
