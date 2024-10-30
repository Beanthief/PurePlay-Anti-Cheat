from sklearn.preprocessing import StandardScaler
import configparser
import listener
import models
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = configparser.ConfigParser()
config.read("config.ini")
batchSize = int(config["Data"]["batchSize"]) # Do I want a globally fixed batch size?
splitRatio = config["Training"]["splitRatio"]
captureKeyboard = config["Listener"]["captureKeyboard"]
captureMouse = config["Listener"]["captureMouse"]
captureController = config["Listener"]["captureController"]
trainingMode = config["Testing"]["trainingMode"]
displayGraph = config["Testing"]["displayGraph"]

inputListener = listener.InputListener(captureKeyboard,
                                       captureMouse,
                                       captureController)
inputListener.start()

# if trainingMode:
#     if captureKeyboard:
#         keyboardModel = models.RNN()
#     if captureMouse:
#         mouseMoveModel = models.RNN()
#         mouseClickModel = models.RNN()
#     if captureController:
#         joystickModel = models.RNN()
#         buttonModel = models.RNN()
#         triggerModel = models.RNN()
# else:
#     if captureKeyboard:
#         keyboardModel = torch.load("models/keyboard.pt")
#     if captureMouse:
#         mouseMoveModel = torch.load("models/mousemove.pt")
#         mouseClickModel = torch.load("models/mouseclick.pt")
#     if captureController:
#         joystickModel = torch.load("models/joystick.pt")
#         buttonModel = torch.load("models/button.pt")
#         triggerModel = torch.load("models/trigger.pt")

scaler = StandardScaler()

def split_data(tensor):
    splitIndex = int(splitRatio * len(tensor))
    trainSplit = tensor[:splitIndex]
    testSplit = tensor[splitIndex:]
    return trainSplit, testSplit

# ADD PLOT_PREDICTIONS EXTERNALLY (the predictions are by batch, not per input)

while True:
    time.sleep(5)

    if trainingMode:
        if captureKeyboard:
            if len(inputListener.keyboardTensor) >= batchSize:
                keyTrain, keyTest = split_data(scaler.fit_transform(inputListener.keyboardTensor)); 
                inputListener.keyboardTensor = inputListener.keyboardTensor[batchSize:]

        if captureMouse:
            if len(inputListener.mouseMoveTensor) >= batchSize:
                moveTrain, moveTest = split_data(scaler.fit_transform(inputListener.mouseMoveTensor)); 
                inputListener.mouseMoveTensor = inputListener.mouseMoveTensor[batchSize:]

            if len(inputListener.mouseClickTensor) >= batchSize:
                clickTrain, clickTest = split_data(scaler.fit_transform(inputListener.mouseClickTensor)); 
                inputListener.mouseClickTensor = inputListener.mouseClickTensor[batchSize:]

        if captureController:
            if len(inputListener.joystickTensor) >= batchSize:
                stickTrain, stickTest = split_data(scaler.fit_transform(inputListener.joystickTensor)); 
                inputListener.joystickTensor = inputListener.joystickTensor[batchSize:]

            if len(inputListener.buttonTensor) >= batchSize:
                buttonTrain, buttonTest = split_data(scaler.fit_transform(inputListener.buttonTensor)); 
                inputListener.buttonTensor = inputListener.buttonTensor[batchSize:]

            if len(inputListener.triggerTensor) >= batchSize:
                triggerTrain, triggerTest = split_data(scaler.fit_transform(inputListener.triggerTensor)); 
                inputListener.triggerTensor = inputListener.triggerTensor[batchSize:]
    else:
        # confidence = 1
        # if captureKeyboard:
        #     with torch.inference_mode():
        #         confidence *= keyboardModel(scaler.fit_transform(inputListener.keyboardTensor))

        # if captureMouse:
        #     with torch.inference_mode():
        #         confidence *= mouseMoveModel(scaler.fit_transform(inputListener.mouseMoveTensor))
        #         confidence *= mouseClickModel(scaler.fit_transform(inputListener.mouseClickTensor))

        # if captureController:
        #     with torch.inference_mode():
        #         confidence *= joystickModel(scaler.fit_transform(inputListener.joystickTensor))
        #         confidence *= buttonModel(scaler.fit_transform(inputListener.buttonTensor))
        #         confidence *= triggerModel(scaler.fit_transform(inputListener.triggerTensor))