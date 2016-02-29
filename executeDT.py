'''
Authors : Apurwa Dandekar, Swati Bhartiya
'''
import pickle
#import trainDT
import sys
import csv
import matplotlib.pyplot as plt
dataspace=[]
noOfCorrect=0
noOfInCorrect=0
profit=0
profitMatrix = [[0 for row in range(4)] for row in range(4)]
confusionMatrix = [[0 for row in range(4)] for row in range(4)]
itemList=["Bolt", "Nut", "Ring" ,"Scrap"]
class1=[]
class2=[]
class3=[]
class4=[]
classPlot1=[] 
classPlot2=[] 
classPlot3=[]
classPlot4=[]
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
'''
This is used to read the .csv file
'''		
def readFile(file):
	global rowsInFile, dataspace
	rowsInFile=[]
	f = open(file, 'rt')
	reader = csv.reader(f)
	#read the file and store each row in a list
	iterate=0
	for row in reader:
		if(len(row)!=0):
			rowsInFile.append(row)
			point=Point(float(row[0]),float(row[1]),float(row[2]))
			dataspace.append(point)
'''
This is used to classify each point 
in the dataspace
'''		
def classify(dataspace,tree):
	global noOfCorrect, noOfInCorrect,profit,confusionMatrix,profitMatrix
	recognition=0
	for point in dataspace:
		row=[point.x,point.y]
		root=tree
		yint=int(point.output)
		while(root.left!=None and root.right!=None):
			attribute=root.attribute
			if(attribute=="x"):
				if(point.x<root.value):
					root=root.left
				else:
					root=root.right
			else:
				if(point.y<root.value):
					root=root.left
				else:
					root=root.right
		if(root.classification==point.output):
			recognition=recognition+1
		profit=profit+profitMatrix[int(root.classification)-1][yint-1]
		count=confusionMatrix[yint-1][int(root.classification)-1]
		confusionMatrix[yint-1][int(root.classification)-1]=count+1
		testResult(root.classification,row)
	rate=(recognition/len(dataspace))*100
	print("Recognition rate:",rate)
	print("Profit:", profit)
	print("Confusion Matrix:")
	print("Bolts Nuts Ring Scrap <- Assigned class")
	print("                         Actual class")
	for i in range(0,4):
		for j in range(0,4):
			if(j==3):
				print( confusionMatrix[i][j], "     ", itemList[i])
			else:
				print( confusionMatrix[i][j], "     ", end="")
		print()
'''
This method is used to differentiate 
the dataset
'''	
def testResult(obtainedClass,input):
	global class1, class2, class3, class4
	if(obtainedClass==1):
		class1.append(input)
	elif(obtainedClass==2):
		class2.append(input)
	elif(obtainedClass==3):
		class3.append(input)
	elif(obtainedClass==4):
		class4.append(input)
'''
This is used to plot the dataset in the
graph
'''		
def plotTestData(tree):
	plt.figure()
	plt.axis([0,1,0,1])
	plt.xlabel("X axis")
	plt.ylabel("Y axis")
	plt.title("Green: Class1, Red: Class2, Blue: Class3, Yellow: Class4")
	for value in class1:
		plt.plot(value[0],value[1],'go')
	plt.hold(True)
	for value in class2:
		plt.plot(value[0],value[1],'ro')
	plt.hold(True)
	for value in class3:
		plt.plot(value[0],value[1],'bo')
	plt.hold(True)
	for value in class4:
		plt.plot(value[0],value[1],'yo')
	plotRegion(tree)
	for value in classPlot1:
		plt.plot(value[0],value[1],'g.',ms=3.0)
	plt.hold(True)
	for value in classPlot2:
		plt.plot(value[0],value[1],'r.', ms=3.0)
	plt.hold(True)
	for value in classPlot3:
		plt.plot(value[0],value[1],'b.', ms=3.0)
	plt.hold(True)
	for value in classPlot4:
		plt.plot(value[0],value[1],'y.', ms=3.0)
	plt.grid(True)
	plt.show()
'''
This is used to plot the regions
'''	
def plotRegion(tree):
	x=0.0
	
	while(x<1.01):
		y=0.0
		while(y<1.01):
			tempList=[]
			x1 = x
			x2 = y
			row=[x1, x2]
			row=[x,y]
			root=tree
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
			classifyPlot(root.classification,row)
			y=y+0.01
		x=x+0.01
'''
This is used to distinguish the dataset 
into different classes
'''
def classifyPlot(obtainedClass,input):
	global classPlot1, classPlot2, classPlot3, classPlot4
	if(obtainedClass==1):
		classPlot1.append(input)
	elif(obtainedClass==2):
		classPlot2.append(input)
	elif(obtainedClass==3):
		classPlot3.append(input)
	elif(obtainedClass==4):
		classPlot4.append(input)
'''
The main program
'''		
def main():
	global profitMatrix
	treeFile=sys.argv[2]
	print(treeFile)
	pklfile = open(treeFile, 'rb')
	tree = pickle.load(pklfile)
	profitMatrix = [[20, -7, -7, -7], [-7, 15, -7, -7], [-7, -7, 5, -7], [-3, -3, -3, -3]]
	readFile(sys.argv[1])
	classify(dataspace, tree)
	plotTestData(tree)

main()

	