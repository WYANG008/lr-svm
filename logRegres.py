
from numpy import *

#load input trainning data set
#input file should be a txt file in below format
#x{0},y{0},label{0}
#x{1},y{1},label{1}
#...
#x{m},y{m},label{m}
#label belongs to {1,0}
def loadDataSet(inputfile):
    trainingSet = []; trainingLabels = []
    fr = open(inputfile)
   
    for line in fr.readlines():
		currLine = line.strip().split('\t')
		dimension=len(currLine)-1
		lineArr =[]
		for i in range(dimension):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[dimension]))
    return trainingSet,trainingLabels


##Sigmoid function
def sigmoid(inX):
    return 1.0/(1+exp(-inX))


#Weights trainning process
#Stochastic Gradient Ascwending
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteraton
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


#Test classifier
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


# trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
def classifierTest(testfile,weights):
  
   
    errorCount = 0; numTestVec = 0.0
    frTest = open(testfile)
   
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        dimension=len(currLine)-1
        for i in range(dimension):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), weights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

