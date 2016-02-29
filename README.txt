README :

1. trainMLP.py takes as input a train_data.csv file.
2. It is executed trainMLP.py as : python trainMLP.py train_data.csv
3. It generates 5 .csv files containing weights after 0, 10, 100, 1000, 10000 epochs named as weights0.csv, weights1.csv, weights2.csv, weights3.csv, weights4.csv
4. It also generates a learning curve that is used to plot the SSD on the Y-axis and Epoch on the X-axis.
5. After executing the trainMLP.py, the executeMLP.py is executed as : python executeMLP.py test_data.csv weights0.csv.
6. On changing the weights.csv file, the executeMLP.py generates different confusion matrices, the profit obtained, the recognition rates on the command line as output.
7. It also creates a plot of the regions along with data points.
8. trainDT.py takes input a train_data.csv file.
NOTE : The number of nodes in the hidden layer can be changed by changing the value in the 'numberOfHiddenNodes' variable from the main method.
9. It is executed trainDT.py as : python trainDT.py train_data.csv
10. It generates the number of nodes, the number of leaf nodes, the maximum depth, the minimum depth, the average depth of the original and pruned tree as output on the command line.
11. It generates as output a dataRoot.pkl file and dataPruned.pkl that have the original tree and the pruned tree as output.
12. It also generates a decision tree graph.
13. executeDT.py takes as input the test_data.csv file.
14. After executing the trainDT.py, the executeDT.py is executed as : python executeDT.py test_data.csv dataRoot.pkl
15. This generates the recognition rate, the profit obtained, the confusion matrix as output on the command line after using the unpruned tree.
16. It also plots a graph of the regions obtained of the data samples.
17. Similarly, if executeDT.py is executed as : python executeDT.py test_data.csv dataPruned.pkl
18. This generates the recognition rate, the profit obtained, the confusion matrix as output on the command line after using the pruned tree.
19. It also plots a graph of the regions obtained of the data samples.
