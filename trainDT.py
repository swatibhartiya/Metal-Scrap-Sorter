'''
Authors : Apurwa Dandekar, Swati Bhartiya
'''
import sys
import csv
import math
import pickle
import copy
import matplotlib.pyplot as plt

x = []
y = []
output = []
rowsInFile = []
xmidpoint = []
ymidpoint = []
dataspace = []
listOfThreshold=[]

rightSplitPoints=[]
leftSplitPoints=[]
downSplitPoints=[]
upSplitPoints=[]
infoGain = []
parentList = []
count=0
chiThres1 = [6.635, 9.210, 11.345]
chiThres5 = [3.841, 5.991, 7.815]
listOfDepth=[]
'''
This is the node present on each node
of the decision tree
'''
class TreeNode:
	left = None
	right = None
	children = []
	value = 0.0 # mid-point value
	attribute = None
	classification = 0
	sampleLength = 0
	parent=None
	# either x or y based on split
	counts = []
	pruned = False
'''
This is used to represent each sample
from the .csv file
'''	
class Point:
	def __init__(self,x,y,output):
		self.x=x
		self.y=y
		self.output=output
'''
This is used to read the .csv file
'''		
def readFile(file):
	global rowsInFile
	rowsInFile=[]
	f = open(file, 'rt')
	reader = csv.reader(f)
	#read the file and store each row in a list
	for row in reader:
		rowsInFile.append(row)
		x.append(float(row[0]))
		y.append(float(row[1]))
		output.append(float(row[2]))
		point=Point(float(row[0]),float(row[1]),float(row[2]))
		dataspace.append(point)
'''
This method is used to calculate the midpoints
between consecutive samples after sorting
the sample based on the attribute
'''		
def calculateMidpoints(list, type):
	temp = []
	for i in range(len(list)):
		if (type == 'x'):
			midpoint = 0.0
			if(i+1 >= len(list)):
				break
			else:
				midpoint = (list[i].x + list[i+1].x)/2
				temp.append(midpoint)
		elif (type == 'y'):
			midpoint = 0.0
			if(i+1 >= len(list)):
				break
			else:
				midpoint = (list[i].y + list[i+1].y)/2
				temp.append(midpoint)
	return temp
'''
This is the decision tree learning
algorithm
'''
def DTL(pointsList):
	global count
	same = True
	classification = 0
	if(len(pointsList)==1):
		classification=pointsList[0].output # if only there is one sample in the pointsList
	else:		
		for idx in range(len(pointsList)): # to check each point belongs to same class
			if(idx != 0):
				if(pointsList[idx].output == classification):
					continue
				else:
					same = False
					break
			classification=pointsList[idx].output
		
	#Terminating condition
	if(same == True):
		node = TreeNode()
		node.classification = classification
		# node.children = pointsList
		node.counts = getClassCounts(pointsList)
		node.sampleLength = len(pointsList) 
		return node
	else:
		best = chooseNode(pointsList) # to choose the best mid-point with the max info gain
		count=count+1
		#print(best.children[0])
		best.left=DTL(best.children[0])
		best.left.parent=best
		best.right=DTL(best.children[1])
		best.right.parent=best
	
	return best

'''
To get the count of the classes for data sample list
'''	
def getClassCounts(listOfPoints):
	count1 = 0
	count2 = 0
	count3 = 0
	count4 = 0
	
	countSizes = []
	
	for point in listOfPoints:
		if(point.output == 1):
			count1 = count1 + 1
		elif (point.output == 2):
			count2 = count2 + 1
		elif (point.output == 3):
			count3 = count3 + 1
		elif (point.output == 4):
			count4 = count4 + 1
	
	countSizes.append(count1)
	countSizes.append(count2)
	countSizes.append(count3)
	countSizes.append(count4)
	
	return countSizes
'''
To calculate the best attribute meant for splitting
'''			
def chooseNode(pointsList):
	global listOfThreshold
	parent=[]
	x = {}
	y = {}
	maxInfoGain = 0
	parent=pointsList
	xSort = sortOnX(parent)
	ySort = sortOnY(parent)
	midpointListX=calculateMidpoints(xSort,"x")
	midpointListY=calculateMidpoints(ySort,"y")
	xDict = splitOnmidPointsX(midpointListX, parent)
	#print("xdict:", xDict)
	x = max(xDict, key = xDict.get)
	igx = xDict[x]
	yDict = splitOnmidPointsY(midpointListY, parent)
	#print("ydict:", yDict)
	y = max(yDict, key = yDict.get)
	igy = yDict[y]
	
	attribute=""
	if(igx > igy):
		maxInfoGain = igx
		attribute="x"
	else:
		maxInfoGain = igy
		attribute="y"
	
	node=TreeNode()
	node.attribute=attribute
	node.counts = getClassCounts(pointsList)
	
	if(attribute == "x"):
		leftSplits, rightSplits = splitPointsOnX(x, parent)
		node.value = x
	else:
		leftSplits, rightSplits= splitPointsOnY(y, parent)
		node.value = y
	node.children = [leftSplits,rightSplits]
	node.sampleLength = len(pointsList) 
	listOfThreshold.append(node)
	return node
	

'''
This method is used to plot a region
using all the pixels
'''	
def plotBoxBoundary(root):
	x=0.0
	dx=0.01
	plt.axis([0,1,0,1])
	while(x<1.01):
		y=0.0
		while(y<1.01):
			class1=plotValue(root,x,y)
			class2=plotValue(root,x+dx,y)
			class3=plotValue(root,x,y+dx)
			if(class1!=class2):
				plt.plot(x,y+(dx/2), 'r.')
			if(class1!=class3):
				plt.plot(x+(dx/2),y, 'r_')
			y=y+0.01
		x=x+0.01
	plt.show()
'''
This method is used to classify each 
pixel in the plot
'''	
def plotValue(root,x,y):
	while(root.left!=None and root.right!=None):
		attribute=root.attribute
		if(attribute=="x"):
			if(x<root.value):
				root=root.left
			else:
				root=root.right
		else:
			if(y<root.value):
					root=root.left
			else:
				root=root.right
	return root.classification
'''
To split the list on mid-point on X attribute
'''			
def splitOnmidPointsX(midpointList,parentPoints):
	IGX = {}
	for midPoint in midpointList:
		splitPointsOnX(midPoint, parentPoints)
		entropy1=calculateEntropy(leftSplitPoints)
		entropy2=calculateEntropy(rightSplitPoints)
		entropyParent=calculateEntropy(parentPoints)
		informationGain=entropyParent-((len(leftSplitPoints)/len(parentPoints)) * entropy1 + (len(rightSplitPoints)/len(parentPoints)) * entropy2)
		IGX[midPoint] = informationGain
	return IGX
'''
To split the list on mid-point on Y attribute
'''		
def splitOnmidPointsY(midpointList,parentPoints):
	IGY = {}
	for midPoint in midpointList:
		splitPointsOnY(midPoint, parentPoints)
		entropy1=calculateEntropy(leftSplitPoints)
		entropy2=calculateEntropy(rightSplitPoints)
		entropyParent=calculateEntropy(parentPoints)
		informationGain=entropyParent-((len(leftSplitPoints)/len(parentPoints)) * entropy1 + (len(rightSplitPoints)/len(parentPoints)) * entropy2)
		IGY[midPoint] = informationGain
	return IGY	
'''
To calculate the entropy of a list
'''	
def calculateEntropy(subList):
	global infoGain
	count1 = 0
	count2 = 0
	count3 = 0
	count4 = 0
	total = len(subList)
	for point in subList:
		if(point.output == 1):
			count1 = count1 + 1
		elif (point.output == 2):
			count2 = count2 + 1
		elif (point.output == 3):
			count3 = count3 + 1
		elif (point.output == 4):
			count4 = count4 + 1
	entropy =0
	if(count1!=0):
		entropy=entropy+(-((count1/total) * math.log(count1/total, 2)))
	if(count2!=0):
		entropy=entropy+(-((count2/total) * math.log(count2/total, 2)))
	if(count3!=0):
		entropy=entropy+(-((count3/total) * math.log(count3/total, 2)))
	if(count4!=0):
		entropy=entropy+(-((count4/total) * math.log(count4/total, 2)))
	return entropy
	
'''
To sort the list on X attribute
'''	
def sortOnX(list):	
	sortedList = (sorted(list, key = lambda point:point.x))
	# for point in sortedList:
		# print(point.x, point.y, point.output)
	return sortedList	
'''
To sort the list on Y attribute
'''		
def sortOnY(list):
	sortedList = (sorted(list, key = lambda point:point.y))
	# for point in sortedList:
		# print(point.x, point.y, point.output)
	return sortedList
'''
Used to split the data samples on the basis of
X attribute
'''	
def splitPointsOnX(midPoint, parentPoints):
	global leftSplitPoints,rightSplitPoints
	leftSplitPoints=[]
	rightSplitPoints=[]
	for point in parentPoints:
		if(point.x < midPoint):
			leftSplitPoints.append(point)
		else:
			rightSplitPoints.append(point)
	return leftSplitPoints, rightSplitPoints
'''
Used to split the data samples on the basis of
Y attribute
'''		
def splitPointsOnY(midPoint,parentPoints):
	global leftSplitPoints,rightSplitPoints
	leftSplitPoints=[]
	rightSplitPoints=[]
	for point in parentPoints:
		if(point.y < midPoint):
			leftSplitPoints.append(point)
		else:
			rightSplitPoints.append(point)
	return leftSplitPoints, rightSplitPoints
'''
Used to traverse the tree inorder
'''
def treeTraversal(root):
	# if (root.left == None and root.right==None):
		# return
	if(root is not None):
		treeTraversal(root.left)
		treeTraversal(root.right)
'''
Used to traverse the tree in preorder
'''
def treePreTraversal(root):
	if(root is not None):
		treeTraversal(root.left)
		treeTraversal(root.right)

'''
To calculate the depth the tree
'''			
def findDepth(root,depth):
	if (root.left == None and root.right==None):
		return depth
	return max(findDepth(root.left,depth+1),findDepth(root.right,depth+1))
'''
To calculate the chi value at the node
'''
def chiValueCompute(node, type):
	pLeft = []
	pRight = []
	delta = 0.0
	pLeft.append(node.counts[0] * node.left.sampleLength/(node.sampleLength))
	pLeft.append(node.counts[1] * node.left.sampleLength/(node.sampleLength))
	pLeft.append(node.counts[2] * node.left.sampleLength/(node.sampleLength))
	pLeft.append(node.counts[3] * node.left.sampleLength/(node.sampleLength))
	XL = 0
	temp = 0
	for idx in range(len(node.left.counts)):
		if(pLeft[idx] != 0):
			temp = math.pow((node.left.counts[idx]  - pLeft[idx]),2)
			temp = temp / pLeft[idx]
			XL = XL + temp
	
	pRight.append(node.counts[0] * node.right.sampleLength/(node.sampleLength))
	pRight.append(node.counts[1] * node.right.sampleLength/(node.sampleLength))
	pRight.append(node.counts[2] * node.right.sampleLength/(node.sampleLength))
	pRight.append(node.counts[3] * node.right.sampleLength/(node.sampleLength))
	XR = 0
	temp = 0
	for idx in range(len(node.right.counts)):
		if(pRight[idx] != 0):
			temp = math.pow((node.right.counts[idx]  - pRight[idx]),2)
			temp = temp / pRight[idx]
			XR = XR + temp
		
	delta = XL + XR		
	dof = 0
	for i in range(len(node.counts)):
		if(node.counts[i] != 0):
			dof = dof + 1
	
	if(type == 1):
		if(delta > chiThres1[dof-2]):
			print('prune', node.sampleLength)
			if(len(node.children[0])>len(node.children[1])):
				node.classification = node.left.classification
			else:
				node.classification = node.right.classification
			node.left = None
			node.right = None
			
	else:
		if(delta > chiThres5[dof-2]):
			if(len(node.children[0])>len(node.children[1])):
				node.classification = node.left.classification
			else:
				node.classification = node.right.classification
			node.left = None
			node.right = None
	
'''
To calculate the average depth of the tree
'''	
def averageDepth(root,depth):
	global listOfDepth
	if (root.left == None and root.right==None):
		listOfDepth.append(depth)
		return depth
	return max(averageDepth(root.left,depth+1),averageDepth(root.right,depth+1))

'''
Used to calculate the chi squared value 
for the nodes in the tree
'''
def calculateChiSquare(node, pruningDepth, presentDepth, type):
	if(presentDepth == pruningDepth):
		if (node.left is not None and node.right is not None):
			if(node.left.left == None and node.right.right == None and node.left.right == None and node.right.left == None):
				chiValueCompute(node, type)
				return
	
	if (node.left is not None and node.right is not None):
		calculateChiSquare(node.left, pruningDepth, presentDepth + 1, type)
		calculateChiSquare(node.right, pruningDepth, presentDepth + 1, type)

def getLNoOfLeaf(tree):   
	if(tree == None):
		return 0 
	if(tree.left ==None and tree.right==None):      
		return 1           
	else:
		return getLNoOfLeaf(tree.left)+ getLNoOfLeaf(tree.right)

def getLNoOfNodes(tree):   
	if(tree == None):
		return 0 
	if(tree.left ==None and tree.right==None):      
		return 1           
	else:
		return getLNoOfNodes(tree.left)+ getLNoOfNodes(tree.right)+1 		
'''
USed to calculate the min depth of the tree
'''
def minDepth(root):
    if (root.left == None and root.right==None):
        return 0
    if (root.left == None and root.right == None):
       return 1
    if (root.left==None):
       return minDepth(root.right) + 1
    if (root.right==None):
       return minDepth(root.left) + 1
    return min(minDepth(root.left), minDepth(root.right)) + 1;
'''
This is the main program
'''
def main():
	global xmidpoint, ymidpoint
	readFile(sys.argv[1])
	root = DTL(dataspace)

	plotBoxBoundary(root)
	outputRoot = open('dataRoot.pkl','wb')
	pickle.dump(root, outputRoot)

	depth = findDepth(root, 0)
	
	print("Max depth of original tree:", depth)
	count = depth - 1
	noOfLeaf=getLNoOfLeaf(root)
	print("No. of leaf nodes of original tree:",noOfLeaf)
	noOfNodes=getLNoOfNodes(root)
	print("No. of nodes of original tree:",noOfNodes)
	minDe=minDepth(root)
	print("Min depth of original tree: ",minDe)
	averageDepth(root,0)
	sum=0
	for depth in listOfDepth:
		sum=sum+depth
	avaerage=sum/len(listOfDepth)
	print("Average depth of original tree: ",avaerage)
	
	
	while(count != 0):
		calculateChiSquare(root, count, 0, 5)
		count = count - 1

	depth = findDepth(root, 0)
	print("Max depth of pruned tree:", depth)
	noOfLeaf=getLNoOfLeaf(root)
	print("No. of leaf nodes of pruned tree:",noOfLeaf)
	noOfNodes=getLNoOfNodes(root)
	print("No. of nodes of pruned tree:",noOfNodes)
	minDe=minDepth(root)
	print("Min depth of pruned tree: ",minDe)
	averageDepth(root,0)
	sum=0
	for depth in listOfDepth:
		sum=sum+depth
	avaerage=sum/len(listOfDepth)
	print("Average depth of pruned tree: ",avaerage)

	plotBoxBoundary(root)
	outputPruned = open('dataPruned.pkl','wb')
	pickle.dump(root, outputPruned)

main()