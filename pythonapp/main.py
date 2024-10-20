import configparser
import listener
import sklearn
import pandas
import models
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = configparser.ConfigParser()
config.read("config.ini")
dataDirectory = config["Data"]["dataDirectory"]
dataType = config["Data"]["dataType"] # Check if this works
splitRatio = config["Data"]["splitRatio"]
writeDelay = config["Listener"]["writeDelay"]
captureKeyboard = config["Listener"]["captureKeyboard"]
captureMouse = config["Listener"]["captureMouse"]
captureController = config["Listener"]["captureController"]
outputToFile = config["Listener"]["outputToFile"]
displayGraph = config["Testing"]["displayGraph"]

inputListener = listener.InputListener(writeDelay, 
                                       captureKeyboard, 
                                       captureMouse, 
                                       captureController, 
                                       outputToFile)
inputListener.start()

encoder = sklearn.preprocessing.LabelEncoder()
scaler = sklearn.preprocessing.StandardScaler()

# Is this really how I want to output? RNN vs FNN etc.
# If I output this way, the files and matrix will grow indefinitely.
# An alternative would be to only send the inputs once the matrix elements
# reach a fixed batch size. This would allow me to clear the elements each time.
# Would moderation want access to the files?

while(True):
    if outputToFile:
        for file in os.listdir(dataDirectory):
            path = os.path.join(dataDirectory, file)
            data = pandas.read_csv(path, header=None)
            data = data.apply(encoder.fit_transform)
            data = scaler.fit_transform(data)
            tensor = torch.tensor(data, dtype=dataType).to(device)
            model = models.RNN(inputCount=3, 
                    nodeCount=3, 
                    layerCount=3, 
                    sequenceLength=2, 
                    classCount=2).to(device)
            splitIndex = int(splitRatio * len(tensor))
            trainSplit = tensor[:splitIndex]
            testSplit = tensor[splitIndex:]
            if displayGraph:
                model.plotPredictions("Untrained Model", trainSplit, testSplit)
            model.train()
            if displayGraph:
                model.plotPredictions("Trained Model", trainSplit, testSplit)
    else:
        tensor = torch.tensor()
        for device, input in inputListener.inputMatrix:
            #tensor.append?
