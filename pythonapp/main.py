import configparser
import listener
import sklearn
import models
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = configparser.ConfigParser()
config.read("config.ini")
dataDirectory = config["Data"]["dataDirectory"]
batchSize = config["Data"]["batchSize"] # Do I want a globally fixed batch size?
splitRatio = config["Training"]["splitRatio"]
captureKeyboard = config["Listener"]["captureKeyboard"]
captureMouse = config["Listener"]["captureMouse"]
captureController = config["Listener"]["captureController"]
displayGraph = config["Testing"]["displayGraph"]

inputListener = listener.InputListener(captureKeyboard, 
                                       captureMouse, 
                                       captureController)
inputListener.start()

# Next step
keyboardModel = models.RNN()
mouseMoveModel = models.RNN()
mouseClickModel = models.RNN()
joystickModel = models.RNN()
buttonModel = models.RNN()
triggerModel = models.RNN()

encoder = sklearn.preprocessing.OneHotEncoder()
scaler = sklearn.preprocessing.StandardScaler()

def formatData(tensor):
    encodedTensor = encoder.fit_transform(tensor)
    scaledTensor = scaler.fit_transform(encodedTensor)
    return scaledTensor

def splitData(tensor):
    splitIndex = int(splitRatio * len(tensor))
    trainSplit = tensor[:splitIndex]
    testSplit = tensor[splitIndex:]
    return trainSplit, testSplit

while True:
    time.sleep(10)
    inputListener.save_to_file()

    if captureKeyboard:
        if len(inputListener.keyboardTensor) >= batchSize:
            keyTrain, keyTest = splitData(formatData(inputListener.keyboardTensor)); 
            inputListener.keyboardTensor = inputListener.keyboardTensor[batchSize:]
            # Next step

    if captureMouse:
        if len(inputListener.mouseMoveTensor) >= batchSize:
            moveTrain, moveTest = splitData(formatData(inputListener.mouseMoveTensor)); 
            inputListener.mouseMoveTensor = inputListener.mouseMoveTensor[batchSize:]

        if len(inputListener.mouseClickTensor) >= batchSize:
            clickTrain, clickTest = splitData(formatData(inputListener.mouseClickTensor)); 
            inputListener.mouseClickTensor = inputListener.mouseClickTensor[batchSize:]

    if captureController:
        if len(inputListener.joystickTensor) >= batchSize:
            stickTrain, stickTest = splitData(formatData(inputListener.joystickTensor)); 
            inputListener.joystickTensor = inputListener.joystickTensor[batchSize:]

        if len(inputListener.buttonTensor) >= batchSize:
            buttonTrain, buttonTest = splitData(formatData(inputListener.buttonTensor)); 
            inputListener.buttonTensor = inputListener.buttonTensor[batchSize:]

        if len(inputListener.triggerTensor) >= batchSize:
            triggerTrain, triggerTest = splitData(formatData(inputListener.triggerTensor)); 
            inputListener.triggerTensor = inputListener.triggerTensor[batchSize:]
