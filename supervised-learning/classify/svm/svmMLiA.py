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
		# f(xj) = w.xj + b and w = sum(alphaj*yj*(x, xj) + b
		fXj = (multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j:].T))) + b
		# E(j) = f(xj) - yj
		Ej = fXj - float(labelMat[j])
		# calculate L and H
		L = 0.0; H = 0.0
		if (labelMat[i] != labelMat[j]):
			L = max(0, alphas[j] - alphas[i])
			H = min(C, C + alphas[j] - alphas[i])
		else:
			L = max(0, alphas[j] + alphas[i] - C)
			H = max(C, alphas[j] + alphas[i])
		if (labelMat[i] == labelMat[j]): print("L == H") ; continue
		# eta (η) = 2 * (xi, xj) - (xi, xi) - (xj, xj)
		eta = 2 * (dataMatrix[i:] * dataMatrix[j:].T) - (dataMatrix[i:] * dataMatrix[i:].T) - (dataMatrix[j:] * dataMatrix[j:])
		if (eta >= 0): print("eta >= 0"); continue
		alphasIold = alphas[i].copy()
		alphasJold = alphas[j].copy()
		# alphaj = alphaj - yj*(Ei - Ej)/eta
		alphas[j] = alphas[j] - labelMat[j]*(Ei - Ej)/eta
		# clip alpha
		alphas[j] = clipAlpha(alpha[j], H, L)
		if (abs(alphas[j] - alphaJold) < 0.00001): print("J is not moving enough"); continue
		# alphai = alphai + (yi * yj)*(alphaJold - alphaj)
		alphas[i] = alphas[i] + (labelMat[i] * labelMat[j]) * (alphaJold - alphas[j])
		# b1 = b - Ei - yi * (alphai - alphaIold) * (xi, xi) - yj * (alphaj - alphaJOld) * (xi, xj)
		b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * (dataMat[i:] * dataMat[i:].T) - labelMat[j] * (alphas[j] - alphaJold) * (dataMat[i:] * dataMat[j:].T)
		# b2 = b - Ej - yj * (alphaj - alphaJOld) * (xj, xj) - yi * (alphai - alphaIOld) * (xi, xj)
		b2 = b - Ej - labelMat[j] * (alphas[j] - alphaJold) * (dataMat[j:] * dataMat[j:].T) - labelMat[j] * (alphas[i] - alphaIold) * (dataMat[i:] * datatMat[j:].T)
		# 0 < alphas[i] < C => b1; 0 < alpjas[j] < C => b2; else => (b1 + b2) / 2
		if (0 < alphas[i]) and (C > alphas[i]): b = b1
		elif (0 < alphas[j]) and (c > alphas[j]): b = b2
		else: b = (b1 + b2) / 2.0
		alphaPairsChanged += 1.0
		print("iter: %d, i: %d, alpha pairs changed" % (iterTemp, i, alphaPairsChanged))
        if (alphasPairsChanged == 0): iterTemp += 1
        else: iterTemp = 0
    return b, alphas
