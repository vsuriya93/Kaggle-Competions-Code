import os
import sys
from numpy import *
from scipy import io
import matplotlib.pyplot as plt
from pybrain.structure import *
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
data = io.loadmat('ex4data1.mat')
X = data['X']
[m, n] = shape(X)
Y = data['y']
Y = reshape(Y, (len(Y), -1))
numLabels = len(unique(Y))
Y[Y == 10] = 0
X = hstack((ones((m, 1)), X))
n = n+1
nInput = n
nHidden0 = int(n / 5)
nOutput = numLabels
inLayer = LinearLayer(nInput)
hiddenLayer = SigmoidLayer(nHidden0)
outLayer = SoftmaxLayer(nOutput)

net = FeedForwardNetwork()
net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)

theta1 = FullConnection(inLayer, hiddenLayer)
theta2 = FullConnection(hiddenLayer, outLayer)

net.addConnection(theta1)
net.addConnection(theta2)

net.sortModules()

def convertToOneOfMany(Y):
    rows, cols = shape(Y)
    numLabels = len(unique(Y))
    Y2 = zeros((rows, numLabels))
    for i in range(0, rows):
        Y2[i, Y[i]] = 1
    return Y2

allData = SupervisedDataSet(n, numLabels)
Y2 = convertToOneOfMany(Y)

allData.setField('input', X)
allData.setField('target', Y2)

train = BackpropTrainer(net, dataset=dataTrain, learningrate=0.1, momentum=0.1)
trueTrain = dataTrain['target'].argmax(axis=1)
trueTest = dataTest['target'].argmax(axis=1)

EPOCHS = 20
for i in range(EPOCHS):
    train.trainEpochs(1)
    outTrain = net.activateOnDataset(dataTrain)
    outTrain = outTrain.argmax(axis=1)
    resTrain = 100 - percentError(outTrain, trueTrain)

    outTest = net.activateOnDataset(dataTest)
    outTest = outTest.argmax(axis=1)
    resTest = 100 - percentError(outTest, trueTest)

    print("epoch: %4d " % train.totalepochs,"\ttrain acc: %5.2f%% " % resTrain, "\ttest acc: %5.2f%%" % resTest)

prediction = net.activate(X1)
print(prediction)
p = argmax(prediction, axis=0)
print("predicted output is \t" + str(p))
