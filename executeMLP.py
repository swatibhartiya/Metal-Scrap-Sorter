'''
Authors : Apurwa Dandekar, Swati Bhartiya
'''
import sys
import csv
import math
import matplotlib.pyplot as plt
'''
Global variables meant to be used for the program
'''
rowsInDataFile = []
rowsInWeightsFile = []
floatWeights = []
inputNodes=[]
hiddenNodes=[]
outputNodes=[]
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
This class is used to create each node in the 
neural network
'''
class Node:
	input = 0
	weights = []
	a = 0
	delta = 0
	y = 0
	def __init__(self,number):
		self.number = number
'''
This method is used to read the dataFile passed 
as input from the command line and store the 
data samples into the rowsInFile list
'''		
def readFile(file):
	global rowsInDataFile, rowsInWeightsFile
	
	if(file == sys.argv[1]):
		i = 0
		f = open(file, 'rt')
		reader = csv.reader(f)
		#read the file and store each row in a list
		for row in reader:
			if(len(row)!=0):
				rowsInDataFile.append(row)
				i = i+1
			else:
				break
	else:
		f = open(file, 'rt')
		reader = csv.reader(f)
		#read the file and store each row in a list
		for row in reader:
			rowsInWeightsFile.append(row)
		
		for r in rowsInWeightsFile:
			temp=[]
			for i in r:
				i = float(i)
				temp.append(i)
			floatWeights.append(temp)
'''
This method is used to compute the summation of the 
weights and the activation of the node
'''				
def summation(nodeNumber,incomingNodes):
	sum = 0
	for node in incomingNodes:
		sum = sum+(node.a )*( node.weights[nodeNumber])
	return sum
'''
This method is used to compute the sigmoid of the 
summation of the weights with the activation
'''		
def calculateHwx(value):
	sigmoid = 1/(1+math.exp(-value))
	return sigmoid
'''
This method is used to run the neural network 
with the weights from the input .csv file
'''
def runNetwork():
	for row in rowsInDataFile:
		tempList=[]
		x1 = float(row[0])
		x2 = float(row[1])
		y = float(row[2])
		tempList=[x1, x2]
		#initialize value for input nodes
		for node, r in zip(inputNodes, row):
			if(node.number == 2):
				node.a = 1
			else:
				node.a = float(r)
				
		#update values for hiddenNodes
		for hiddenNode in hiddenNodes:
			if(hiddenNode.number != (len(hiddenNodes)-1)):
				hiddenNode.input = summation(hiddenNode.number,inputNodes)
				hiddenNode.a = calculateHwx(hiddenNode.input)
			else:
				hiddenNode.a=1
				
		#update values for outputNode		
		for outputNode in outputNodes:
				outputNode.input=summation(outputNode.number,hiddenNodes)
				outputNode.a = calculateHwx(outputNode.input)
		testResult(outputNodes, y, tempList)
	rate=noOfCorrect/len(rowsInDataFile)
	print("Recognition rate:", rate*100)
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
		
	plotTestData()
'''
This method is used to classify the data from
the input .csv file and it calculates the profit
and generates a confusion matrix 
'''				
def testResult(outputNode,y,inputValues):
	global noOfCorrect, noOfInCorrect,profit,confusionMatrix
	max=-111111111
	maxNode=5555
	yint=int(y)
	for node in outputNode:
		if(node.a>max):
			max=node.a
			maxNode=node.number #output node represents the class
	if(maxNode==y-1):
		noOfCorrect=noOfCorrect+1
	else:
		noOfInCorrect=noOfInCorrect+1
	classify(maxNode,inputValues)
	profit=profit+profitMatrix[maxNode][yint-1]
	count=confusionMatrix[yint-1][maxNode]
	confusionMatrix[yint-1][maxNode]=count+1
'''
This method is used to generate the graph 
depicting the regions to the data samples
belong'''	
def plotTestData():
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
	plotRegion()
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
	
"""
plots the classification region for each
"""
def plotRegion():
	x=0.0
	
	while(x<1.01):
		y=0.0
		while(y<1.01):
			tempList=[]
			x1 = x
			x2 = y
			row=[x1, x2]
			#initialize value for input nodes
			for node, r in zip(inputNodes, row):
				if(node.number == 2):
					node.a = 1
				else:
					node.a = float(r)
					
			#update values for hiddenNodes
			for hiddenNode in hiddenNodes:
				if(hiddenNode.number != (len(hiddenNodes)-1)): # -1 for bias node
					hiddenNode.input = summation(hiddenNode.number,inputNodes)
					hiddenNode.a = calculateHwx(hiddenNode.input)
				else:
					hiddenNode.a=1 # for bias node
					
			#update values for outputNode		
			for outputNode in outputNodes:
					outputNode.input=summation(outputNode.number,hiddenNodes)
					outputNode.a = calculateHwx(outputNode.input)
			testPlot(outputNodes,row)
			y=y+0.01
		x=x+0.01
		
		
"""
to classify each pixel in the graph
"""
def testPlot(outputNodes,inputValues):
	max=-111111111
	maxNode=5555
	for node in outputNodes:
		if(node.a>max):
			max=node.a
			maxNode=node.number
	classifyPlot(maxNode,inputValues)
	
"""
to classify each pixel and put it into the list for plotting
"""
def classifyPlot(obtainedClass,input):
	global classPlot1, classPlot2, classPlot3, classPlot4
	if(obtainedClass==0):
		classPlot1.append(input)
	elif(obtainedClass==1):
		classPlot2.append(input)
	elif(obtainedClass==2):
		classPlot3.append(input)
	elif(obtainedClass==3):
		classPlot4.append(input)

"""
to classify each point of the csv and put it into the list for plotting
"""		
def classify(obtainedClass,input):
	global class1, class2, class3, class4
	if(obtainedClass==0):
		class1.append(input)
	elif(obtainedClass==1):
		class2.append(input)
	elif(obtainedClass==2):
		class3.append(input)
	elif(obtainedClass==3):
		class4.append(input)
	
def main():
	global profitMatrix
	readFile(sys.argv[1]) # test.csv
	readFile(sys.argv[2]) # weights.csv
	
	profitMatrix = [[20, -7, -7, -7], [-7, 15, -7, -7], [-7, -7, 5, -7], [-3, -3, -3, -3]]

	numberOfHiddenNodes = 5
# create network
	for i in range(0,3):
		node=Node(i)
		node.weights=floatWeights[i]
		inputNodes.append(node)
		
	#create hidden  nodes with random weights
	for i in range(0,numberOfHiddenNodes + 1): # +1 for bias node
		node=Node(i)
		node.weights=floatWeights[i+3]
		hiddenNodes.append(node)
		
	#create output nodes with random weights
	for i in range(0,4):
		node=Node(i)
		outputNodes.append(node)
	runNetwork()

main()