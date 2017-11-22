# lr-svm
This is python implementation of logistic regression and support vector machines

#for logistic regression testing

the weightages training method is stochastic gradient ascendent
two files are provided, one is training data named 'lrTrain.txt',one is 'lrTest.txt'
in each line, there are 22 numbers, the first 21 numbers are the data vectors, while the last one is the label {1,0}

Please follow the below steps to test this program

1. load required libray

from numpy import *
import logRegres

2. load training data

trainingSet,trainingLabels=logRegres.loadDataSet('lrTrain.txt')

3. training weights

weights=logRegres.stocGradAscent1(array(trainingSet),trainingLabels,150)

4. test classifier error using the testing files

logRegres.classifierTest('lrTest.txt',weights)



#for svm testing

the optimization method used is full Platt SMO algorithm
the kernel applied is Gaussian Kernel

two files are provided, one is training data named 'svmTrain.txt', one is 'svmTest.txt'
In each line, there are three numbers, the first 2 are data points coordinates, while the last one is the label {-1,1}

Please follow the below to test this program

import svmMLiA
svmMLiA.testRbf('svmTrain.txt','svmTest.txt')
