from sklearn.preprocessing import StandardScaler
import configparser
import listener
import keyboard
import models
import pandas
import torch
import time
import os

config = configparser.ConfigParser()
config.read("config.ini")
programMode = int(config["General"]["programMode"])                   # 0 = Data Collection, 1 = Model Training, 2 = Live Analysis
captureKeyboard = int(config["Collection"]["captureKeyboard"])
captureMouse = int(config["Collection"]["captureMouse"])
captureController = int(config["Collection"]["captureController"])
dataDirectory = config["Collection"]["dataDirectory"]
dataLabel = config["Collection"]["dataLabel"]                         # control, cheat
killKey = config["Collection"]["killKey"]
dataType = config["Training"]["dataType"]
modelDirectory = config["Training"]["modelDirectory"]
pollInterval = int(config["Analysis"]["pollInterval"])                # Time between batch predictions (window size)
displayGraph = int(config["Analysis"]["displayGraph"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputListener = listener.InputListener(captureKeyboard, captureMouse, captureController, (0))
inputListener.start()

def convert_to_tensor(list, shape): # Use during analysis
    return torch.tensor(list) if list else torch.empty(shape)

match programMode:

    ########## Data Collection ##########
    case 0:
        while True:
            time.sleep(10)
            if keyboard.is_pressed(killKey):
                while keyboard.is_pressed(killKey):
                    time.sleep(0.5)
                inputListener.buttonData = inputListener.buttonData[:-2] # Strip kill key from data
                inputListener.save_to_files(dataDirectory, dataLabel)
                break
            inputListener.save_to_files(dataDirectory, dataLabel)

    ########## Model Training ##########
    case 1:
        buttonLSTM = models.LSTMClassifier(inputSize=4,
                                           hiddenSize=32,
                                           classCount=2,
                                           layerCount=2,
                                           device=device).to(device)
        if captureMouse:
            moveLSTM = models.LSTMClassifier(inputSize=3,
                                             hiddenSize=32,
                                             classCount=2,
                                             layerCount=2,
                                             device=device).to(device)
        if captureController:
            stickLSTM = models.LSTMClassifier(inputSize=4,
                                              hiddenSize=32,
                                              classCount=2,
                                              layerCount=2,
                                              device=device).to(device)
            triggerLSTM = models.LSTMClassifier(inputSize=3,
                                                hiddenSize=32,
                                                classCount=2,
                                                layerCount=2,
                                                device=device).to(device)

        buttonTensor = torch.empty((0, 4))  # Button input (4 features)
        moveTensor = torch.empty((0, 3))    # Mouse movement (3 features)
        stickTensor = torch.empty((0, 4))   # Stick input (4 features)
        triggerTensor = torch.empty((0, 3)) # Trigger input (3 features)

        for fileName in os.listdir(dataDirectory):
            filePath = os.path.join(dataDirectory, fileName)
            if os.path.isfile(filePath) and fileName.endswith(".csv"):
                dataFrame = pandas.read_csv(filePath)
                tensor = torch.tensor(dataFrame.values, dtype=dataType)
                
                if "cheat" in fileName: # Decide on how to pass yTrain
                    if "button" in fileName:
                        bTensor = torch.cat((buttonTensor, tensor), dim=0)
                    elif "move" in fileName:
                        mTensor = torch.cat((moveTensor, tensor), dim=0)
                    elif "stick" in fileName:
                        sTensor = torch.cat((stickTensor, tensor), dim=0)
                    elif "trigger" in fileName:
                        tTensor = torch.cat((triggerTensor, tensor), dim=0)
                elif "control" in fileName:
                    if "button" in fileName:
                        bTensor = torch.cat((buttonTensor, tensor), dim=0)
                    elif "move" in fileName:
                        mTensor = torch.cat((moveTensor, tensor), dim=0)
                    elif "stick" in fileName:
                        sTensor = torch.cat((stickTensor, tensor), dim=0)
                    elif "trigger" in fileName:
                        tTensor = torch.cat((triggerTensor, tensor), dim=0)

        if bTensor.size(0) > 0:
            buttonLSTM.train() # Add appropriate training params
            torch.save(buttonLSTM.state_dict(), f"{modelDirectory}/button.pt")
        if mTensor.size(0) > 0:
            moveLSTM.train() # Add appropriate training params
            torch.save(moveLSTM.state_dict(), f"{modelDirectory}/move.pt")
        if sTensor.size(0) > 0:
            stickLSTM.train() # Add appropriate training params
            torch.save(stickLSTM.state_dict(), f"{modelDirectory}/stick.pt")
        if tTensor.size(0) > 0:
            triggerLSTM.train() # Add appropriate training params
            torch.save(triggerLSTM.state_dict(), f"{modelDirectory}/trigger.pt")

    ########## Live Analysis ##########
    case 2: 
        scaler = StandardScaler()
        buttonLSTM = torch.load(f"{modelDirectory}/buttonLSTM.pt")
        if captureMouse:
            moveLSTM = torch.load(f"{modelDirectory}/moveLSTM.pt")
        if captureController:
            stickLSTM = torch.load(f"{modelDirectory}/stickLSTM.pt")
            triggerLSTM = torch.load(f"{modelDirectory}/triggerLSTM.pt")
        while True: # Or while game is running?
            time.sleep(pollInterval)
            confidence = 1
            with torch.inference_mode():
                output = buttonLSTM(scaler.fit_transform(inputListener.buttonData))
                confidence *= torch.softmax(output)[1] # Verify output format before doing this
                inputListener.buttonData = []
            if captureMouse:
                with torch.inference_mode():
                    output = moveLSTM(scaler.fit_transform(inputListener.moveData))
                    confidence *= torch.softmax(output)[1] # Verify output format before doing this
                    inputListener.moveData = []
            if captureController:
                with torch.inference_mode():
                    output = stickLSTM(scaler.fit_transform(inputListener.stickData))
                    confidence *= torch.softmax(output)[1] # Verify output format before doing this
                    inputListener.stickData = []
                    output = triggerLSTM(scaler.fit_transform(inputListener.triggerData))
                    confidence *= torch.softmax(output)[1] # Verify output format before doing this
                    inputListener.triggerData = []
            print(confidence)
            # Display graph?
            # Encrypt output?

inputListener.stop()