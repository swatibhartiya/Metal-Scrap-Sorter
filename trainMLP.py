'''
Authors : Apurwa Dandekar, Swati Bhartiya
'''
import sys
import random
import csv
import math
import matplotlib.pyplot as plt
'''
Global variables meant to be used for the program
'''
layer1 = [[-2 for row in range(5)] for row in range(3)]
layer2 = [[-2 for row in range(4)] for row in range(6)]
hiddenLayer = []
outputLayer = []
rowsInFile = []
inputNodes = []
hiddenNodes = []
outputNodes = []
layer = []
sqrRootError = [] # use for graph of SSD v/s epoch
epoch=[]
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
	global rowsInFile
	rowsInFile=[]
	f = open(file, 'rt')
	reader = csv.reader(f)
	#read the file and store each row in a list
	for row in reader:
		rowsInFile.append(row)
'''
This method is used to write the updated weights after
epochs 0, 10, 100, 1000 and 10000 to weights0.csv, 
weights1.csv, weights2.csv, weights3.csv, weights4.csv
respectively
'''			
def writeFile(weightFileName):
	weightsList=[]
	ofile  = open(weightFileName, "w", newline='')
	writer = csv.writer(ofile)
	for inputNode in inputNodes:
		writer.writerow(inputNode.weights)
	for hiddenNode in hiddenNodes:
		writer.writerow(hiddenNode.weights)
'''
This method is used to compute the summation of the 
weights and the activation of the node
'''	
def summation(nodeNumber, incomingNodes):
	sum = 0
	for node in incomingNodes:
		sum =sum+ node.a * node.weights[nodeNumber]
	return sum
'''
This method is used to compute the summation of the 
weights and the delta for each node in the network
'''	
def weightsSummation(outgoingNodes, hiddenNode):
	sum = 0
	for out, weight in zip(outgoingNodes, hiddenNode.weights):
		sum = sum+(out.delta)*(weight)
	return sum
'''
This method is used to compute the updated weights
for each edge in the network
'''
def updateWeight(currentNode, outgoingNodes):
	alpha = 0.1
	nodeNumber = currentNode.number
	tempWeights = []
	
	for weight, out in zip(currentNode.weights, outgoingNodes):
		tempWeight = weight + (alpha * out.delta * currentNode.a)
		tempWeights.append(tempWeight)
	return tempWeights
'''
This method is used to compute the sigmoid of the 
summation of the weights with the activation
'''	
def calculateHwx(value):
	sigmoid = 1/(1+math.exp(-value))
	return sigmoid

'''
This method is used to pass through the network 
each data sample from the CSV file and update weights
based on the error propagated back to it
'''	
def backPropagation():
	global sqrRootError, inputNodes, hiddenNodes, outputNodes
	error=[]
	
	for row in rowsInFile:
		x1 = row[0]
		x2 = row[1]
		y = float(row[2])
		#initialize value for input nodes
		for node, r in zip(inputNodes, row):
			if(node.number == 2):
				node.a = 1
			else:
				node.a = float(r)
				
		#update values for hiddenNodes
		for hiddenNode in hiddenNodes:
			if(hiddenNode.number != (len(hiddenNodes) - 1)): # 5
				hiddenNode.input = summation(hiddenNode.number, inputNodes)
				hiddenNode.a = calculateHwx(hiddenNode.input)
			else:
				hiddenNode.a=1  # for bias node
				
		#update values for outputNode		
		for outputNode in outputNodes:
				outputNode.input=summation(outputNode.number,hiddenNodes)
				outputNode.a = calculateHwx(outputNode.input)
				#for the output check
				if(y == outputNode.number + 1):
					outputNode.y = 1
				else:
					outputNode.y = 0
					
				
		# calculate error
		tempError=0
		for outputNode in outputNodes:
			absoluteValue=abs(outputNode.y-outputNode.a)
			tempError=tempError + absoluteValue
		square = tempError*tempError
		error.append(square)
			
		#calculate delta values for outputNodes 
		for outputNode in outputNodes:
			outputNode.delta=(outputNode.a)*(1-outputNode.a)*(outputNode.y-outputNode.a)
		
		#calculate delta values for hiddenNodes 
		for hiddenNode in hiddenNodes:
			if(hiddenNode.number!= (len(hiddenNodes) - 1)): # 5
				hiddenNode.delta=(hiddenNode.a)*(1-hiddenNode.a) * weightsSummation(outputNodes,hiddenNode)
		
		#update weight for hiddennodes
		for hiddenNode in hiddenNodes:
			hiddenNode.weights = updateWeight(hiddenNode, outputNodes)
			
		#update weight for inputNodes
		for inputNode in inputNodes:
			inputNode.weights = updateWeight(inputNode, hiddenNodes)

	sumOfSquares = 0
	for e in error:
		sumOfSquares = sumOfSquares + e
	
	sqrRoot = math.sqrt(sumOfSquares)
	sqrRootError.append(sqrRoot)
	
"""plot epoch vs error graph"""	
def plotGraph():
	plt.figure()
	global epoch,sqrRootError
	errorRange2=sqrRootError[0]
	lenghthOFepochList=len(epoch)
	errorRange1=sqrRootError[0]
	plt.xlabel("Epoch")
	plt.ylabel("Error")
	plt.plot(epoch, sqrRootError, 'yo')
	plt.axis([1, lenghthOFepochList, min(sqrRootError), max(sqrRootError)])
	plt.show()
'''
The main program from where the network is created
and the data read in is used to train the network
'''
def main():
	global epoch
	readFile(sys.argv[1])
	numberOfHiddenNodes = 5
	#create inputNodes with random weights
	for i in range(0,3):
		node=Node(i)
		temp=[]
		for j in range(0, numberOfHiddenNodes):
			randomValue=random.uniform(-1,1)
			# randomValue=0.1
			temp.append(randomValue)
		node.weights=temp
		inputNodes.append(node)
		
	#create hidden nodes with random weights
	for i in range(0,numberOfHiddenNodes + 1):  # +1 for bias node
		node=Node(i)
		temp=[]
		for j in range(0,4):
			randomValue=random.uniform(-1,1)
			# randomValue=0.1
			temp.append(randomValue)
		node.weights=temp
		hiddenNodes.append(node)
		
	#create output nodes with random weights
	for i in range(0,4):
		node=Node(i)
		outputNodes.append(node)
	
	noOfIterations=0
	while(noOfIterations<10001):
		backPropagation()
		if(noOfIterations==0):
			print("noOfIterations:", noOfIterations)
			writeFile("weights0.csv")
		elif(noOfIterations==10):
			print("noOfIterations:", noOfIterations)
			writeFile("weights1.csv")
		elif(noOfIterations==100):
			print("noOfIterations:", noOfIterations)
			writeFile("weights2.csv")
		elif(noOfIterations==1000):
			print("noOfIterations:", noOfIterations)
			writeFile("weights3.csv")
		elif(noOfIterations==10000):
			print("noOfIterations:", noOfIterations)
			writeFile("weights4.csv")
		epochValue=noOfIterations+1
		epoch.append(epochValue)
		noOfIterations=noOfIterations+1
	plotGraph()
main()