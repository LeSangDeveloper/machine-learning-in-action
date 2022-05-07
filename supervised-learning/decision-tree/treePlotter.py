import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeText, centerPt, parentPt, nodeType):
	createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot():
	fig = plt.figure(1, facecolor="white")
	fig.clf()
	createPlot.ax1 = plt.subplot(111, frameon=False)
	plotNode('decision node', (0.1, 0.5), (0.2, 0.4), decisionNode)
	plotNode('leaf node', (1, 1), (2, 2), leafNode)
	plt.show()

def getNumLeafs(myTree):
	numLeafs = 0
	rootNodeKey = list(myTree.keys())[0]
	dictSubTrees = myTree[rootNodeKey]
	for _, value in dictSubTrees.items():
		if type(value).__name__ == 'dict':
			numLeafs += getNumLeafs(value)
		else: numLeafs+= 1
	return numLeafs

def getTreeDepth(myTree):
	maxDepth = 0
	rootNodeKey = list(myTree.keys())[0]
	dictSubTrees = myTree[rootNodeKey]
	for _, value in dictSubTrees.items():
		thisDepth = 0
		if type(value).__name__ == 'dict':
			thisDepth += 1 + getTreeDepth(value)
		else: thisDepth = 1
		if thisDepth > maxDepth: maxDepth = thisDepth
	return maxDepth

def retrieveTree(i):
	listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers':
                    {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': 
                    {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
	return listOfTrees[i]

def plotMidText(centerPt, parentPt, txtString):
	xMid = (centerPt[0] + parentPt[0])/2.0 + centerPt[0]
	yMid = (centerPt[1] + parentPt[1])/2.0 + centerPt[1]
	createPlot.ax1.text(xMid, yMid, txtString)

 
def plotTree(myTree, parentPt, nodeTxt):
	numLeafs = getNumLeafs(myTree)
	getTreeDepth(myTree)
	centerPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
	plotMidText(centerPt, parentPt, nodeTxt)
	rootNodeTxt = list(myTree.keys())[0]
	plotNode(rootNodeTxt, centerPt, parentPt, decisionNode)
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	dictSubTrees = myTree[rootNodeTxt]
	for key, value in dictSubTrees.items():
		if type(value).__name__=='dict':
			plotTree(value, centerPt, str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(value, (plotTree.xOff, plotTree.yOff), centerPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), centerPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	plotTree.totalD = float(getTreeDepth(inTree))
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.xOff = -0.5/plotTree.totalW
	plotTree.yOff = 1.0
	plotTree(inTree, (0.5, 1.0), '')
	plt.show()

def classify(inputTree, featLabels, testVec):
	rootNodeKey = list(inputTree.keys())[0]
	subTrees = inputTree[rootNodeKey]
	featIndex = featLabels.index(rootNodeKey)
	for key, value in subTrees.items():
		if testVec[featIndex] == key:
			if type(value).__name__=='dict':
				return classify(value, featLabels, testVec)
			else: return value
