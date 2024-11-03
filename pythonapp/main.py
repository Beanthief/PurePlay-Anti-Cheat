from sklearn.preprocessing import StandardScaler
import configparser
import listener
import models
import torch
import time

config = configparser.ConfigParser()
config.read("config.ini")
batchSize = int(config["Data"]["batchSize"]) # Do I want a globally fixed batch size?
captureKeyboard = bool(config["Listener"]["captureKeyboard"])
captureMouse = bool(config["Listener"]["captureMouse"])
captureController = bool(config["Listener"]["captureController"])
pollInterval = int(config["Runtime"]["pollInterval"])
trainingMode = bool(config["Training"]["trainingMode"])
isCheatingData = bool(config["Training"]["isCheatingData"])
displayGraph = bool(config["Testing"]["displayGraph"])

inputListener = listener.InputListener(captureKeyboard, captureMouse, captureController)
inputListener.start()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if trainingMode:
    buttonLSTM = models.LSTM()
    #buttonTransformer = models.Transformer()
    if captureMouse:
        moveLSTM = models.LSTM()
        #moveTransformer = models.Transformer()
    if captureController:
        stickLSTM = models.LSTM()
        #stickTransformer = models.Transformer()
        triggerLSTM = models.LSTM()
        #triggerTransformer = models.Transformer()
else:
    buttonLSTM = torch.load("pythonapp/models/buttonLSTM.pt")
    buttonTransformer = torch.load("pythonapp/models/buttonTransformer.pt")
    if captureMouse:
        moveLSTM = torch.load("pythonapp/models/moveLSTM.pt")
        moveTransformer = torch.load("pythonapp/models/moveTransformer.pt")
    if captureController:
        stickLSTM = torch.load("pythonapp/models/stickLSTM.pt")
        stickTransformer = torch.load("pythonapp/models/stickTransformer.pt")
        triggerLSTM = torch.load("pythonapp/models/triggerLSTM.pt")
        triggerTransformer = torch.load("pythonapp/models/triggerTransformer.pt")

scaler = StandardScaler()

while True: # Or while game is running?
    time.sleep(pollInterval)
    if trainingMode:
        break
    else:
        confidence = 1
        if len(inputListener.buttonTensor) >= batchSize:
            with torch.inference_mode():
                confidence *= buttonLSTM(scaler.fit_transform(inputListener.buttonTensor[:batchSize]))
                confidence *= buttonTransformer(scaler.fit_transform(inputListener.buttonTensor[:batchSize]))
                inputListener.buttonTensor = inputListener.buttonTensor[batchSize:]
        
        if captureMouse and len(inputListener.moveTensor) >= batchSize:
            with torch.inference_mode():
                confidence *= moveLSTM(scaler.fit_transform(inputListener.moveTensor[:batchSize]))
                confidence *= moveTransformer(scaler.fit_transform(inputListener.moveTensor[:batchSize]))
                inputListener.moveTensor = inputListener.moveTensor[batchSize:]

        if captureController:
            if len(inputListener.stickTensor) >= batchSize:
                with torch.inference_mode():
                    confidence *= stickLSTM(scaler.fit_transform(inputListener.stickTensor[:batchSize]))
                    confidence *= stickTransformer(scaler.fit_transform(inputListener.stickTensor[:batchSize]))
                    inputListener.stickTensor = inputListener.stickTensor[batchSize:]
            if len(inputListener.triggerTensor) >= batchSize:
                with torch.inference_mode():
                    confidence *= triggerLSTM(scaler.fit_transform(inputListener.triggerTensor[:batchSize]))
                    confidence *= triggerTransformer(scaler.fit_transform(inputListener.triggerTensor[:batchSize]))
                    inputListener.triggerTensor = inputListener.triggerTensor[batchSize:]
        print(confidence)
        # Display graph?
                                                     
# Train on loaded tensors
if isCheatingData:
    yTrain = [batchSize, 1]
else:
    yTrain = [batchSize, 0]