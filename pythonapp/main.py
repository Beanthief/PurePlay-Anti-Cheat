import torch
import pandas
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keyData = pandas.read_csv("kb.csv", header=None)
keyTensor = torch.tensor(keyData.values, dtype=torch.float64).to(device)
print(keyTensor)

moveData = pandas.read_csv("mmove.csv", header=None)
moveTensor = torch.tensor(moveData.values, dtype=torch.float64).to(device)
print(moveTensor)

clickData = pandas.read_csv("mclick.csv", header=None)
clickTensor = torch.tensor(clickData.values, dtype=torch.float64).to(device)
print(clickTensor)

keySplit = int(0.8 * len(keyTensor))
moveSplit = int(0.8 * len(moveTensor))
clickSplit = int(0.8 * len(clickTensor))

keyTrain = keyTensor[:keySplit]
keyTest = keyTensor[keySplit:]
moveTrain = moveTensor[:moveSplit]
moveTest = moveTensor[moveSplit:]
clickTrain = clickTensor[:clickSplit]
clickTest = clickTensor[clickSplit:]

keyModel = models.RNN(inputCount=3, 
                      nodeCount=3, 
                      layerCount=3, 
                      sequenceLength=2, 
                      classCount=2).to(device)

moveModel = models.RNN(inputCount=3, 
                      nodeCount=3, 
                      layerCount=3, 
                      sequenceLength=2, 
                      classCount=2).to(device)

clickModel = models.RNN(inputCount=5, 
                      nodeCount=3, 
                      layerCount=3, 
                      sequenceLength=2, 
                      classCount=2).to(device)

models.plotPredictions(keyModel, keyTrain, keyTest)
models.plotPredictions(moveModel, moveTrain, moveTest)
models.plotPredictions(clickModel, clickTrain, clickTest)
